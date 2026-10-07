package main

import "testing"

func TestAdapterFlagsRejectDuplicateAndMalformedMappings(t *testing.T) {
	var ids adapterFlags
	for _, value := range []string{"missing", "=path", "id="} {
		if err := ids.Set(value); err == nil {
			t.Fatal(value)
		}
	}
	if err := ids.Set("support=/local/path=with-equals"); err != nil {
		t.Fatal(err)
	}
	if ids["support"] != "/local/path=with-equals" {
		t.Fatal(ids)
	}
	if err := ids.Set("support=other"); err == nil {
		t.Fatal("duplicate silently replaced head")
	}
}
