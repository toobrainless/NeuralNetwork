English text follows below.

% Версия 1.0 (апрель 2015 года)
% Автор — Данил Фёдоровых (http://www.hse.ru/staff/df)

Руководство по использованию корпоративного стиля презентации НИУ ВШЭ в пакете Beamer.

1. Для использования шрифтов Myriad Pro в соответствии с брендбуком Высшей школы экономики, эти шрифты нужно установить в систему. Инструкцию по установке шрифтов для ОС Windows 7 можно найти здесь: http://windows.microsoft.com/ru-ru/windows7/install-or-delete-fonts , для Mac OS — здесь: https://support.apple.com/ru-ru/HT201749 .

2. Для правильной компиляции документов нужно использовать движок XeLaTeX. Этот движок входит в современные дистрибутивы LaTeX (MikTeX, TeX Live), процедура его выбора зависит от конкретной программы-редактора (некоторые программы выберут XeLaTeX для компиляции документа автоматически, прочитав первую строку %!TEX TS-program = xelatex). Если вы раньше не работали с XeLaTeX, прочитайте краткую статью о его особенностях: http://ru.wikipedia.org/wiki/XeTeX . XeLaTeX — современный движок, который весьма прост в использовании; документы, сделанные для другого движка (например, pdfTeX) практически не нужно будет модифицировать для работы с новым движком.

3. Для создания презентаций рекомендуется модифицировать под свои нужды файл example-beamer-HSE.tex — в нем настроены все параметры для правильного отображения корпоративной темы. 


% Version 1.0 (April 2015)
% Danil Fedorovykh (http://www.hse.ru/en/staff/df)

Higher School of Economics official Beamer theme user manual.

1. In order to use font styles specified in HSE brand book, you should install Myriad Pro fonts. For Windows 7 installation instruction, see http://windows.microsoft.com/en-us/windows7/install-or-delete-fonts . For Mac OS, see: https://support.apple.com/en-us/HT201749 .

2. The document should be compiled in XeLaTeX. this engine comes with LaTeX distributions (MikTeX, TeX Live) and can be chosen in your .tex editor settings. (In some editors, it will be chosen automatically). XeLaTeX is modern easy to use, only minimal amendments to your existing pdfTeX documents will be required. If you never worked with XeLaTeX before, read the Wikipedia page: http://en.wikipedia.org/wiki/XeTeX.

3. In the file example-beamer-HSE-en.tex, all settings necessary for compilation are given in the preamble. It is better to modify this file for your own needs rather than to try to integrate your existing Beamer files with HSE theme.