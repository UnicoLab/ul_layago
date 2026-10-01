package layago

import (
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const selfModule = "github.com/UnicoLab/ul_layago"

// allowedImport keeps the engine extractable: the standard library, the
// layago packages themselves, and golang.org/x/text and golang.org/x/sys,
// which Hive already depends on. No Hive, pgx or other third-party package.
func allowedImport(path string) bool {
	if path == selfModule || strings.HasPrefix(path, selfModule+"/") {
		return true
	}
	for _, prefix := range []string{"golang.org/x/text/", "golang.org/x/sys/"} {
		if strings.HasPrefix(path, prefix) {
			return true
		}
	}
	first, _, _ := strings.Cut(path, "/")
	return !strings.Contains(first, ".")
}

func TestImportGraphIsSelfContained(t *testing.T) {
	fset := token.NewFileSet()
	checked := 0
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == "testdata" || strings.HasPrefix(d.Name(), ".")) && path != "." {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		f, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		checked++
		for _, imp := range f.Imports {
			p, err := strconv.Unquote(imp.Path.Value)
			if err != nil {
				return err
			}
			if !allowedImport(p) {
				t.Errorf("%s imports %s; internal/layago may import only the standard library, x/text, x/sys and itself", path, p)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if checked < 10 {
		t.Fatalf("only %d files checked; the walk is not seeing the packages", checked)
	}
}
