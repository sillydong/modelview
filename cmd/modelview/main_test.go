package main

import "testing"

func TestOrDash(t *testing.T) {
	if got := orDash(""); got != "-" {
		t.Errorf("orDash(\"\") = %q, want -", got)
	}
	if got := orDash("qwen2"); got != "qwen2" {
		t.Errorf("orDash(qwen2) = %q", got)
	}
}
