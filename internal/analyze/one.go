package analyze

import (
	"context"

	"github.com/sillydong/modelview/internal/model"
)

// One 计算**单个张量**的统计与量化分析，就地填入 tn。
//
// 详情页要的是"这一个张量"，不是整个模型：全量扫描冷跑实测 17.6 秒
// （qwen2.5:3b / 434 张量），而用户点开的是其中一个。
//
// ## 它只写 tn，不写 m
//
// 这条是**并发契约**，不是巧合：`Analyze` 会顺带把缓存里的结果填进
// m 的**所有**张量，而界面是在另一个 goroutine 里跑这条扫描的 ——
// 主线程同时在渲染张量列表（用户随时可能按 Esc 回去），
// 那就成了一写一读的数据竞争。One 不碰 m 的任何字段，也不碰缓存，
// 于是调用方只要在**副本**上扫（见 tui 里的 scanCmd），就完全没有共享写。
//
// ## 为什么不读写缓存
//
// 整份模型的缓存是「模型路径 + 采样上限」为键的一份 JSON。
// 往里面写单个张量是**部分写入**，而部分写入正是下次全量扫描时
// "看起来命中了、其实缺一半"的来源 —— 那类故障不报错，
// 只是数字少了几块。单张量的加速由内存里的 tn.Stats 承担：
// 扫过一次之后 NeedsWork 就是 false，再点开是瞬时的。
//
// 采样上限固定用 DefaultSampleLimit（约 40 MB 样本）。不做成参数：
// 交互式浏览没有调它的场景，而多一个参数就多一处要解释的东西。
// 将来真需要，再加 Options。
func One(ctx context.Context, m *model.Model, tn *model.Tensor) error {
	if !NeedsWork(tn) {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	src, err := openSource(m)
	if err != nil {
		return err
	}
	//nolint:errcheck // 只读源，Close 失败不改变已算出的结果
	defer src.Close()

	return analyzeOne(src, tn, DefaultSampleLimit)
}
