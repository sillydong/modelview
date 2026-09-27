package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/sillydong/modelview/internal/discover"
	"github.com/sillydong/modelview/internal/humanize"
)

// runScan 扫描本机模型目录并输出。
//
// **非 JSON 的那条分支生产路径上已经没有调用方**（无参数时进 TUI 了），
// 但它不是死代码：`TestRunScan_没有模型时也要提未完成的下载` 直接调它，
// 而那条测试守着的是"首次 pull 到一半时，未完成的下载必须被说出来" ——
// 全工具唯一一条会引导破坏性操作的提示。
//
// 第二版注释里还写过"命令行看到的与界面里显示的必须是同一批信息"，
// **那句话是错的**：scanLine 的最后一列是完整路径，TUI 的行里没有路径列。
// 两者共享的是"取原因"（discover.ErrReason）与数字格式化（internal/humanize），
// 不是同一段渲染代码。别拿一句不成立的话当保留理由。
//
// 等 TUI 稳定后再定：删掉非 JSON 分支（连测试一起），还是给它一个 --no-tui。
//
// **异步分层**：Scan 只 stat，瞬时返回，先把首屏打出来；
// 再逐个 Fill 补格式与参数量，每读完一个就地刷新那一行。
// 一次读完再显示的话，HF 缓存里几百个文件会让用户盯着空屏几十秒。
//
// JSON 模式例外：JSON 是一个整体，不能流式拼 —— 那边等全部填完。
func runScan(ctx context.Context, asJSON bool) error {
	res := discover.Scan(ctx, discover.Options{})
	for _, e := range res.Errs {
		fmt.Fprintf(os.Stderr, "modelview: %s\n", e)
	}

	if asJSON {
		res.Items = discover.FillAll(ctx, res.Items)
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(res); err != nil {
			return fmt.Errorf("序列化 JSON: %w", err)
		}
		return nil
	}

	// **未完成的下载要在"没有模型"的早退之前说**。
	//
	// ollama 是下完之后才写 manifest 的（实测：manifest 的时间戳比 blob 晚几秒），
	// 所以"第一次 pull 下到一半"这个最常见的情形里 Items 正好是空的 ——
	// 早退在前的话，12.85 GiB 占着盘而输出一个字都不提，
	// 而这恰恰是这条提示要保护的那个场景。
	if n := len(res.InProgress); n > 0 {
		fmt.Printf("另有 %d 个未完成的下载（合计 %s），不算可回收：\n"+
			"  那是 ollama 正在下载或上次中断留下的，删掉会毁掉下载\n",
			n, humanize.Bytes(totalSize(res.InProgress)))
	}

	if len(res.Items) == 0 {
		fmt.Println("没有发现模型文件。")
		fmt.Println("扫过这些目录（不存在的会被跳过）：")
		for _, p := range discover.Paths() {
			fmt.Printf("  %-12s %s\n", p.Source, p.Dir)
		}
		return nil
	}

	// 先把"有多少个"打出来 —— Fill 要读每个文件的头部，
	// 模型多时要几秒，用户得先知道有东西在跑
	fmt.Printf("发现 %d 个模型", len(res.Items))
	if n := len(res.Orphans); n > 0 {
		fmt.Printf("，另有 %d 个孤儿 blob 可回收 %s",
			n, humanize.Bytes(totalSize(res.Orphans)))
	}
	if len(res.Items) > 1 {
		fmt.Print("（正在读取格式与参数量…）")
	}
	fmt.Println()

	// 读一个、打一行。
	//
	// **不要就地刷新**。早先这里用 `\x1b[1A` 上移一行再重画，想把
	// "先列文件、再补格式"做成原地更新 —— 但数据行有一百多字符，
	// 在 80 列的终端上每行占两三屏行，上移一行落在上一行的中间，
	// 结果是刚打印的那行被擦掉、只剩最后一条。实测（pty 下跑）
	// 输出里能看到一串 `^[[1A^[[J`，而管道输出反而正常，
	// 于是"人看到的"和"脚本看到的"是两回事。
	//
	// 逐行追加没有这个问题，也不需要知道终端宽度。
	for i := range res.Items {
		if ctx.Err() != nil {
			break
		}
		discover.Fill(&res.Items[i])
		fmt.Println(scanLine(res.Items[i]))
	}

	// 孤儿 blob 单独一段：它不属于任何模型，
	// 混在模型列表里会让人以为那也是模型
	if len(res.Orphans) > 0 {
		fmt.Printf("\n孤儿 blob（没有被任何模型引用，可回收 %s）\n",
			humanize.Bytes(totalSize(res.Orphans)))
		for _, o := range res.Orphans {
			fmt.Println(orphanLine(o))
		}
	}
	return nil
}

// scanLine 渲染模型库的一行。
//
// 三段信息按"先看得见的、后读出来的"排列：名字与大小是 Scan 阶段
// 就有的，格式/架构/参数量要 Fill 之后才有 —— 后者没读到时
// 不占位，免得界面上出现一串空白让人以为读失败。
func scanLine(it discover.Item) string {
	// 直接用 Name：discover 的契约是"每个条目都有名字"（见 Item.Name），
	// 由 TestScan_每个条目都有名字 在生产者那侧钉住。
	//
	// 这里原先还有一层 `if Name == "" { 用文件名 }` 的兜底，但它是**死代码**：
	// scanDir 与 findOrphans 都是拿文件名的，ollama 拿的是 model:tag，
	// 没有一条路径会产出空名字 —— 而一段永远不会执行的兜底并不提供保护，
	// 只是让"生产者漏填"这件事看起来已经被处理了。
	out := fmt.Sprintf("  %-28s %-11s %12s  %s",
		humanize.Truncate(it.Name, 28), it.Source, humanize.Bytes(it.Size), it.Path)

	if it.Err != "" {
		// 与 TUI 共用同一个"取原因"的函数：路径在最后一列已经有了
		return out + "  ⚠" + discover.ErrReason(it)
	}
	if it.Format != "" {
		out += "  " + string(it.Format)
	}
	if it.Arch != "" {
		out += " · " + it.Arch
	}
	if it.Params > 0 {
		out += " · " + humanize.Count(it.Params) + " 个参数"
	}
	return out
}

// orphanLine 渲染一行孤儿 blob。
//
// 只显示文件名不显示全路径：全路径里那一长串 sha256 已经把
// 有用信息占满了，而用户关心的是"哪个文件、多大"。
func orphanLine(o discover.Item) string {
	return fmt.Sprintf("  %-72s %12s", filepath.Base(o.Path), humanize.Bytes(o.Size))
}

// totalSize 返回一组条目的字节总数。
func totalSize(items []discover.Item) int64 {
	var n int64
	for _, it := range items {
		n += it.Size
	}
	return n
}
