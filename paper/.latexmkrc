$xelatex = 'xelatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';
$pdf_mode = 5;

add_cus_dep('nlo', 'nls', 0, 'makenlo2nls');
sub makenlo2nls {
    system("makeindex \"$_[0].nlo\" -s nomencl.ist -o \"$_[0].nls\"");
}