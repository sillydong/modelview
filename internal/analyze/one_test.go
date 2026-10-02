package analyze

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// oneModel 造一个三个 F32 张量的模型，都指向同一段数据。
//
// 三个张量指同一段数据是安全的：测试只关心"谁被算了"，不关心数值。
func oneModel(t *testing.T) (*model.Model, string) {
	t.Helper()
	data := make([]byte, 16)
	putF32(data[0:], 1)
	putF32(data[4:], 2)
	putF32(data[8:], 3)
	putF32(data[12:], 4)
	p, dataStart := stFile(t, map[string]any{
		"a": map[string]any{"dtype": "F32", "shape": []int64{4}, "data_offsets": []int64{0, 16}},
	}, data)
	return stModel(t, p, dataStart, 4, "a", "b", "c"), p
}

// One 只动被点名的那个张量。
//
// 这是详情页的性能前提：用户点开一行，不该让整个模型重扫一遍
// （实测全量冷跑 17.6 秒 / 434 个张量）。**它一旦退化成"顺手全扫"，
// 界面上看不出任何区别** —— 只是每次点开都卡十几秒，
// 所以这条必须有测试看着。
func TestOne_只动被点名的张量(t *testing.T) {
	m, _ := oneModel(t)

	if err := One(context.Background(), m, m.Tensors[1]); err != nil {
		t.Fatalf("One 失败: %v", err)
	}
	if m.Tensors[1].Stats == nil {
		t.Fatal("被点名的张量没算出统计")
	}
	if m.Tensors[0].Stats != nil || m.Tensors[2].Stats != nil {
		t.Error("One 顺手扫了别的张量 —— 详情页会因此卡十几秒")
	}
}

// **已经算完的张量不再碰盘** —— 这条契约只能靠 I/O 观测。
//
// 原计划写的是"设一个 Stats 哨兵、断言它没被覆盖"，但那个写法**测不到
// 早退**：Stats != nil 而 F32 且 QuantSims 为空时 NeedsWork 仍返回 true
// （analyze.go 的浮点分支要求 QuantSims 非空），所以 One 照样开文件、
// 解码、补模拟；哨兵没被覆盖只是靠 analyzeOne 自己的 `if tn.Stats == nil`
// 守卫。实测证据：把模型文件挪走，那种写法立刻报 no such file。
//
// 所以这里改成**让模型文件根本不存在**：只有 One 提前返回，
// 才不会去打开它。不提前返回的话 openSource 失败、测试就红。
func TestOne_已算完就不碰盘(t *testing.T) {
	// 路径指向一个不存在的文件 —— 这就是探针
	m := &model.Model{Path: "/nonexistent/x.safetensors", Format: model.FormatSafeTensors}
	tn := &model.Tensor{
		Name: "w", Dims: []int64{64}, Dtype: model.DtypeF32,
		ByteSize: 256, ParamCount: 64,
		Stats: &model.Stats{Count: 999},
		// **必须让 NeedsWork 为假**：浮点类型要 QuantSims 非空才算算完
		QuantSims: []model.QuantSim{{Target: "Q8_0"}},
	}

	if err := One(context.Background(), m, tn); err != nil {
		t.Fatalf("已算完的张量不该再去读文件（应当提前返回）: %v", err)
	}
	if tn.Stats.Count != 999 {
		t.Errorf("已有统计被覆盖了：Count = %d, want 999", tn.Stats.Count)
	}
	if len(tn.QuantSims) != 1 || tn.QuantSims[0].Target != "Q8_0" {
		t.Errorf("已有模拟被覆盖了: %+v", tn.QuantSims)
	}
}

// 目标张量自己失败时错误要带出来 —— 详情页要显示"为什么没统计"。
func TestOne_失败要报错(t *testing.T) {
	m, _ := oneModel(t)
	m.Tensors[0].NonContiguous = true

	err := One(context.Background(), m, m.Tensors[0])
	if err == nil {
		t.Fatal("非连续布局的张量没报错 —— 详情页会显示一个看着正常的错分布")
	}
	if m.Tensors[0].Stats != nil {
		t.Error("失败之后不该留下 Stats")
	}
}

// ctx 已取消时要立刻返回，不读盘。
func TestOne_ctx取消(t *testing.T) {
	m, _ := oneModel(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := One(ctx, m, m.Tensors[0]); err == nil {
		t.Fatal("ctx 已取消却成功返回")
	}
}

// One **完全不碰缓存**。
//
// 断言的是"缓存目录根本没被建出来"，而不是"内容对不对"：
// 单张量扫描往整份模型缓存里写是**部分写入**，
// 而部分写入正是下次全量扫描"看起来命中了、其实缺一半"的来源。
// 缓存相邻于模型文件（<模型目录>/.modelview-cache/），所以这里能直接看。
func TestOne_不写缓存(t *testing.T) {
	m, p := oneModel(t)

	if err := One(context.Background(), m, m.Tensors[0]); err != nil {
		t.Fatalf("One 失败: %v", err)
	}
	dir := filepath.Join(filepath.Dir(p), cacheDirName)
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Errorf("One 建了缓存目录 %s —— 单张量扫描不该碰整份模型缓存", dir)
	}
}
