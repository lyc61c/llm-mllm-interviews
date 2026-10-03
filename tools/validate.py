"""Validate schema, citation edges and generated local Markdown links."""
import re
import sys
from common import ROOT, load_bank, validate
from supplement import validate_supplement


def main():
    sources, questions = load_bank()
    errors = validate(sources, questions)
    errors.extend(validate_supplement(questions, sources))
    for path in ROOT.rglob("*.md"):
        if any(part in {'.git', '.cache', '__pycache__'} for part in path.relative_to(ROOT).parts):
            continue
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).exists():
                errors.append(f"{path.relative_to(ROOT)}: broken local link {target}")
    if errors:
        print("FAIL\n" + "\n".join(errors))
        return 1
    print(f"PASS: {len(questions)} questions, {len(sources)} source records; schema, references, user question coverage and local links valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
