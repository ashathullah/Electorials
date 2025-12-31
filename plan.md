This application reads pdfs and extracts text from them using OCR technology

here are the steps that will be taken to implement this application:
1. extracting image from pdf (PNG format)

2. processing the first page of the pdf using gemini
    these are the information that will be extracted from the first page:
    ```json
    {
        "document_metadata": {
            "language_detected": null,
            "state": null,
            "electoral_roll_year": null,
            "revision_type": null,
            "qualifying_date": null,
            "publicationation_date": null,
            "roll_identification_text": null,
            "total_pages": null,
            "page_number": null
        },

        "constituency": {
            "assembly_constituency_number": null,
            "assembly_constituency_name": null,
            "assembly_reservation_status": null,
            "parliamentary_constituency_number": null,
            "parliamentary_constituency_name": null,
            "parliamentary_reservation_status": null
        },

        "administrative_details": {
            "town_or_village": null,
            "ward_number": null,
            "post_office": null,
            "police_station": null,
            "block_or_taluk": null,
            "subdivision": null,
            "district": null,
            "state": null,
            "pin_code": null
        },

        "part_details": {
            "part_number": null,
            "sections": []
        },

        "polling_station": {
            "polling_station_number": null,
            "polling_station_name": null,
            "polling_station_address": null,
            "polling_station_type": null,
            "auxiliary_polling_station_count": null
        },

        "elector_summary": {
            "serial_number_start": null,
            "serial_number_end": null,
            "male_electors": null,
            "female_electors": null,
            "third_gender_electors": null,
            "total_electors": null
        },

        "authority": {
            "authority_designation": null,
            "signature_present": null
        }
    }
    ```

3. recognizing the language of the pdf based on the information extracted from the first page. based on the language, the following steps will be taken:
    1. skipping first 2 pages if the pdf is in English.
    2. skipping first 3 pages if the pdf is in Tamil.

4. Cropping the voter list from the 3rd page if it is in English or from the 4th page if it is in Tamil.
each page will have 3 columns and each column will have around 10 rows. so the cropped image will be divided into 3 equal parts vertically and each part will be further divided into 10 equal parts horizontally.
    3 x 10 = 30 images for each page.

5. Each cropped part will be processed using OCR to extract text. (Tesseract OCR will be used for this purpose)
6. The extracted text will be cleaned and formatted into a structured format (like CSV or JSON) for easy access and analysis.
5. The extracted text will be structured into a tabular format with the following columns

7. Finally, the structured data will be saved to a csv file and mongodb database for further use with inforamtion gathered from the voter list and the first page of the pdf.