package analyze

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sillydong/modelview/internal/model"
)

// modelFile 造一个假的模型文件（缓存只关心它的大小与 mtime）。
func modelFile(t *testing.T, n int) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "m.gguf")
	if err := os.WriteFile(p, make([]byte, n), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// mkModel 造一个带统计的张量，用于存取。
func mkModel(path string, size int64) *model.Model {
	m := &model.Model{Path: path, FileSize: size}
	m.Tensors = []*model.Tensor{{Name: "w", ParamCount: 4}}
	m.Tensors[0].Stats = &model.Stats{Count: 4, Min: -1, Max: 1, Mean: 0.5, Std: 0.25}
	return m
}

func TestCache_存与取(t *testing.T) {
	p := modelFile(t, 100)
	st, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}

	c := newCache(p, false)
	if err := c.save(mkModel(p, st.Size()), DefaultSampleLimit); err != nil {
		t.Fatalf("save 失败: %v", err)
	}
	if !c.exists() {
		t.Fatalf("写入后 exists 应为 true（路径 %s）", c.path())
	}

	// 换一个全新的 model 对象，从缓存灌回来
	fresh := &model.Model{Path: p, FileSize: st.Size()}
	fresh.Tensors = []*model.Tensor{{Name: "w", ParamCount: 4}}
	if err := c.load(fresh, DefaultSampleLimit); err != nil {
		t.Fatalf("load 失败: %v", err)
	}
	got := fresh.Tensors[0].Stats
	if got == nil {
		t.Fatal("缓存没有灌回 Stats")
	}
	if got.Min != -1 || got.Max != 1 || got.Count != 4 {
		t.Errorf("缓存内容不符: %+v", got)
	}
}

// 缓存写在模型文件旁边，随目录一起搬走。
func TestCache_优先写在模型目录旁(t *testing.T) {
	p := modelFile(t, 100)
	st, _ := os.Stat(p)
	c := newCache(p, false)
	if err := c.save(mkModel(p, st.Size()), DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(filepath.Dir(p), cacheDirName, filepath.Base(p)+".json")
	if c.path() != want {
		t.Errorf("缓存路径 = %q, want %q", c.path(), want)
	}
	if _, err := os.Stat(want); err != nil {
		t.Errorf("缓存文件不存在: %v", err)
	}
}

// patchCacheField 把缓存 JSON 里某个字段改掉，其余原样保留。
//
// 只动目标字段是必须的：如果同时改了文件本身，mtime 也跟着变，
// 于是"大小不符"和"时间不符"两条守卫都会拒绝这条缓存 ——
// 去掉其中任何一条，测试都发现不了。这是本项目反复踩到的
// "多个守卫，测试分不出是谁报的"。
func patchCacheField(t *testing.T, path, old, new string) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("读缓存失败: %v", err)
	}
	patched := strings.Replace(string(raw), old, new, 1)
	if patched == string(raw) {
		t.Fatalf("缓存里没找到 %q，内容: %s", old, raw)
	}
	if err := os.WriteFile(path, []byte(patched), 0o644); err != nil {
		t.Fatal(err)
	}
}

// 文件大小与缓存记录不符时失效 —— 同名不同内容是最容易骗过缓存的情况。
//
// 只改缓存里记的大小，**不动文件本身**，这样 mtime 那一条守卫是满足的，
// 只有大小这一条能拒绝它。
func TestCache_文件大小变化即失效(t *testing.T) {
	p := modelFile(t, 100)
	st, _ := os.Stat(p)
	c := newCache(p, false)
	if err := c.save(mkModel(p, st.Size()), DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}

	patchCacheField(t, c.path(),
		`"file_size":`+strconv.FormatInt(st.Size(), 10),
		`"file_size":`+strconv.FormatInt(st.Size()+1, 10))

	fresh := &model.Model{Path: p, FileSize: st.Size() + 1}
	fresh.Tensors = []*model.Tensor{{Name: "w", ParamCount: 4}}
	if err := c.load(fresh, DefaultSampleLimit); err != nil {
		t.Fatalf("load 不该报错（失效时静默跳过）: %v", err)
	}
	if fresh.Tensors[0].Stats != nil {
		t.Error("记录的大小与文件不符，缓存必须失效")
	}
}

// 文件真的变了（内容与 mtime 都变）同样必须失效。
func TestCache_文件被改写即失效(t *testing.T) {
	p := modelFile(t, 100)
	st, _ := os.Stat(p)
	c := newCache(p, false)
	if err := c.save(mkModel(p, st.Size()), DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(p, make([]byte, 200), 0o644); err != nil {
		t.Fatal(err)
	}
	time.Sleep(10 * time.Millisecond)

	fresh := &model.Model{Path: p, FileSize: 200}
	fresh.Tensors = []*model.Tensor{{Name: "w", ParamCount: 4}}
	if err := c.load(fresh, DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}
	if fresh.Tensors[0].Stats != nil {
		t.Error("文件被改写，缓存必须失效")
	}
}

// 大小相同但 mtime 变了也要失效 —— 只比大小会漏掉"改了内容但长度不变"。
//
// 只改文件的 mtime，不动缓存也不动大小：这样大小那一条守卫是满足的，
// 只有时间这一条能拒绝它。
func TestCache_时间变化即失效(t *testing.T) {
	p := modelFile(t, 100)
	st, _ := os.Stat(p)
	c := newCache(p, false)
	if err := c.save(mkModel(p, st.Size()), DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}

	// 长度不变，只改 mtime
	later := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(p, later, later); err != nil {
		t.Fatal(err)
	}

	fresh := &model.Model{Path: p, FileSize: st.Size()}
	fresh.Tensors = []*model.Tensor{{Name: "w", ParamCount: 4}}
	if err := c.load(fresh, DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}
	if fresh.Tensors[0].Stats != nil {
		t.Error("mtime 变了，缓存必须失效")
	}
}

// schema_version 不匹配要失效：改了 Stats 结构却读到旧缓存会静默得到零值。
//
// 只改缓存里的版本号，大小与 mtime 都保持与文件一致 ——
// 否则另外两条守卫会先拒绝，测试分不出是哪一条起的作用。
func TestCache_结构版本变化即失效(t *testing.T) {
	p := modelFile(t, 100)
	st, _ := os.Stat(p)
	c := newCache(p, false)
	if err := c.save(mkModel(p, st.Size()), DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}

	patchCacheField(t, c.path(),
		`"schema_version":`+strconv.Itoa(schemaVersion), `"schema_version":0`)

	fresh := &model.Model{Path: p, FileSize: st.Size()}
	fresh.Tensors = []*model.Tensor{{Name: "w", ParamCount: 4}}
	if err := c.load(fresh, DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}
	if fresh.Tensors[0].Stats != nil {
		t.Error("schema_version 不匹配，缓存必须失效")
	}
}

// --no-cache 时既不读也不写。
func TestCache_禁用时什么都不做(t *testing.T) {
	p := modelFile(t, 100)
	st, _ := os.Stat(p)
	c := newCache(p, true)
	if err := c.save(mkModel(p, st.Size()), DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}
	if c.exists() {
		t.Error("禁用缓存时不该写出文件")
	}
	if c.path() != "" {
		t.Errorf("禁用时 path 应为空，实际 %q", c.path())
	}

	fresh := &model.Model{Path: p, FileSize: st.Size()}
	fresh.Tensors = []*model.Tensor{{Name: "w"}}
	fresh.Tensors[0].Stats = nil
	if err := c.load(fresh, DefaultSampleLimit); err != nil {
		t.Fatal(err)
	}
	if fresh.Tensors[0].Stats != nil {
		t.Error("禁用时不该读缓存")
	}
}

// 缓存损坏时必须忽略并继续，不能报错也不能崩。
func TestCache_损坏时忽略(t *testing.T) {
	p := modelFile(t, 100)
	c := newCache(p, false)
	if err := os.MkdirAll(filepath.Dir(c.primaryPath()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(c.primaryPath(), []byte("{ 这不是 json"), 0o644); err != nil {
		t.Fatal(err)
	}

	m := &model.Model{Path: p, FileSize: 100}
	m.Tensors = []*model.Tensor{{Name: "w"}}
	if err := c.load(m, DefaultSampleLimit); err != nil {
		t.Fatalf("损坏的缓存应被忽略而不是报错: %v", err)
	}
	if m.Tensors[0].Stats != nil {
		t.Error("损坏的缓存不该灌进数据")
	}
}

// 缺失的缓存不是错误。
func TestCache_文件不存在时静默(t *testing.T) {
	p := modelFile(t, 100)
	c := newCache(p, false)
	m := &model.Model{Path: p, FileSize: 100}
	m.Tensors = []*model.Tensor{{Name: "w"}}
	if err := c.load(m, DefaultSampleLimit); err != nil {
		t.Fatalf("缓存不存在不该报错: %v", err)
	}
}

// 模型文件不存在时 save 静默失败（拿不到 mtime）。
func TestCache_模型文件不存在时save静默(t *testing.T) {
	p := filepath.Join(t.TempDir(), "missing.gguf")
	c := newCache(p, false)
	if err := c.save(mkModel(p, 100), DefaultSampleLimit); err != nil {
		t.Fatalf("save 不该报错: %v", err)
	}
}

// 主缓存目录存在但**不可写**时必须回退到用户缓存目录。
//
// tryMkdir 时代这条会失败：os.MkdirAll 对已存在的目录直接返回 nil，
// 于是只要缓存目录建过一次，就永远解析到主路径、写失败被静默吞掉，
// 从不回退。而"只读挂载下仍然能缓存"正是退路存在的理由。
func TestCache_主目录不可写时回退(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(p, make([]byte, 100), 0o644); err != nil {
		t.Fatal(err)
	}

	// 先把缓存目录建出来（模拟"曾经缓存过"），再设成只读
	cacheDir := filepath.Join(dir, cacheDirName)
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(cacheDir, 0o555); err != nil {
		t.Fatal(err)
	}
	// 收尾时恢复权限，否则 TempDir 清理会失败
	t.Cleanup(func() {
		//nolint:errcheck // 测试清理，失败不影响结论
		os.Chmod(cacheDir, 0o755)
	})

	// HOME 指向临时目录，避免污染真实的用户缓存目录
	t.Setenv("HOME", t.TempDir())

	c := newCache(p, false)
	got := c.path()
	if got == "" {
		t.Fatal("主目录不可写时应有退路，不该返回空")
	}
	if got == c.primaryPath() {
		t.Errorf("主目录不可写，不该仍解析到主路径 %q", got)
	}
	if got != c.fallbackPath() {
		t.Errorf("应回退到 %q，实际 %q", c.fallbackPath(), got)
	}
}
