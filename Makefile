MD = $(wildcard *.md)
ORG = $(wildcard *.org)

book: $(MD)
	jb build ../lecture-notes/

$(MD): %.md: %.org
	pandoc -t markdown-fenced_code_attributes $< -o $@
	sed -i -r 's/@([a-zA-Z0-9_:-]*)/{cite}`\1`/' $@
	sed -i -r 's/\[bibliography:.*\]\(bibliography:.*\)/```{bibliography}\n:style: unsrt\n:filter: docname in docnames\n```/' $@
	sed -i -r 's/\[bibliographystyle:.*\]\(bibliographystyle:.*\)//' $@
	sed -i -r 's/\[file:(.*)\]\((.*)\)/<a href="\2">\1<\/a>/g' $@
	sed -i -r 's/```\{=latex\}/```\{math\}/' $@
	sed -i -r 's/\[(.*)\]\((.*)\.org\)/[\1\]\(\2.md\)/g' $@
	# sed -i -r 's/``` python/```{code-cell} python/' $@
