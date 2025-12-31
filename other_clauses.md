How are we keeping track of which file and which image is processed, let's say I started to crop 1 file and I stopped it at 5th page and again I start it now will it start from 5th page or first page aslo

if I have processed 10/25 files will it start with 11 or 1





$$$


1. Right now we are processing ocr from this directory /extracted/*/images/*.(png/jpg/...)
, now I have changed the directory

/extracted/<file_name>/crops/<page_name>/<images>

2. previously we had only one layer extracted/<file_name>/images
so --limit was for file now we have two layer we need add one more flag to limit the pages from processing.

3. for each page processed the results should be stored in /extracted/<file_name>/output/page_wise/<page_name>.json and the result for the whole file should be in /extracted/<file_name>/output/<file_name>.json

4. I need to know how long it takes to process each image, each page, each file similar to how we implemented in ocr processing.


This project is about gathering information about voters From Pdfs.
Right now all the files are acting separately where I need to Execute Each and every one of them separately To achieve the whole process Now lets organise this code