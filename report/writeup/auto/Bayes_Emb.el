(TeX-add-style-hook
 "Bayes_Emb"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("natbib" "authoryear" "round" "longnamesfirst") ("algpseudocode" "noend") ("fontenc" "T1")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "introduction"
    "method"
    "experiments"
    "discussion"
    "sigkddExp"
    "sigkddExp10"
    "natbib"
    "url"
    "amsmath"
    "amsfonts"
    "amssymb"
    "caption"
    "subcaption"
    "algorithm"
    "algpseudocode"
    "graphicx"
    "listings"
    "fontenc"
    "enumerate"
    "color"
    "relsize"
    "inputenc"
    "multirow")
   (TeX-add-symbols
    "R"
    "Z"
    "Q"
    "C"
    "N"
    "B"
    "bz"
    "bh"
    "bff"
    "bdelta"
    "bsigma"
    "bmu"
    "M"
    "F"
    "E"
    "A"
    "I"
    "U"
    "V"
    "Prob"
    "di"
    "Var")
   (LaTeX-add-environments
    "definition"
    "remark"
    "properties"
    "example"
    "theorem"
    "lemma"
    "corollary"
    "proposition"
    "claim"
    "observation")
   (LaTeX-add-bibliographies
    "Bayes_Em"))
 :latex)

