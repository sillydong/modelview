package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/sillydong/modelview/internal/discover"
)

// runDir 列出**任意目录**下的模型文件。
//
// spec §3 把它与 `modelview scan` 并列：scan 只看已知的十几个模型目录，
// 而用户明确给一个目录时，意图是"就扫这里"。
//
// 目录不存在与"目录里没有模型"必须分开说：后者是正常结果，
// 前者是用户路径写错了 —— 都输出"没有模型"会让人去翻目录。
func runDir(ctx context.Context, dir string, asJSON bool) error {
	st, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if !st.IsDir() {
		return fmt.Errorf("%s 不是目录", dir)
	}

	// **失败判据是 errs 非空，不是 items == nil**：空目录的 items 也是
	// 空切片，拿它当判据会把"这里没有模型"说成"扫描失败"。
	//
	// 注意：ScanDir 现在保证"成功时 items 非 nil"（那是为了让 --json
	// 输出 [] 而不是 null），所以这两条判据在**当前代码里恒等** ——
	// 变异验证把这里改回 `items == nil` 不会有测试变红，那是**等价
	// 变异**而不是测试漏洞。
	//
	// 保留显式的 len(errs)：它不依赖另一个函数顺手做的规范化，
	// 而那条规范化是为了 JSON 形状，不是为了给这里当哨兵。
	items, errs := discover.ScanDir(ctx, dir)
	for _, e := range errs {
		fmt.Fprintf(os.Stderr, "modelview: %s\n", e)
	}
	if len(errs) > 0 {
		return fmt.Errorf("扫描 %s 失败", dir)
	}

	if asJSON {
		items = discover.FillAll(ctx, items)
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(items)
	}
	if len(items) == 0 {
		fmt.Printf("%s 下没有发现模型文件。\n", dir)
		return nil
	}
	fmt.Printf("发现 %d 个模型（正在读取格式与参数量…）\n", len(items))
	items = discover.FillAll(ctx, items)
	for _, it := range items {
		fmt.Println(scanLine(it))
	}
	return nil
}
