This application reads pdfs and extracts text from them using OCR technology

here are the steps that will be taken to implement this application:
1. extracting image from pdf (PNG format)

2. processing the first page of the pdf using gemini
    these are the information that will be extracted from the first page:
    ```json
    {
        "document_metadata": {
            "language_detected": [
            "Tamil",
            "English"
            ],
            "state": "Tamil Nadu",
            "electoral_roll_year": 2025,
            "revision_type": "Special Summary Revision",
            "qualifying_date": "01-01-2025",
            "publication_date": "06-01-2025",
            "roll_type": "Supplement 1",
            "roll_identification": "Special Summary Revision 2025",
            "total_pages": 55,
            "page_number_current": 1
        },
        "constituency_details": {
            "assembly_constituency_number": 114,
            "assembly_constituency_name": "திருப்பூர் (தெற்கு)",
            "assembly_reservation_status": null,
            "parliamentary_constituency_number": 18,
            "parliamentary_constituency_name": "திருப்பூர் (பொது)",
            "parliamentary_reservation_status": null,
            "part_number": 1
        },
        "administrative_address": {
            "town_or_village": "திருப்பூர்",
            "ward_number": "27",
            "post_office": "திருப்பூர் வடக்கு",
            "police_station": "திருப்பூர் வடக்கு வேலாம்பாளையம்",
            "taluk_or_block": "திருப்பூர் (மா)",
            "subdivision": null,
            "district": "திருப்பூர்",
            "pin_code": "641602"
        },
        "part_and_polling_details": {
            "sections": [
            {
                "section_number": "1",
                "section_name": "திருப்பூர் (மா), முருங்கப்பாளையம் 1 தெரு வார்டு எண்.27"
            },
            {
                "section_number": "2",
                "section_name": "திருப்பூர் (மா), முருங்கப்பாளையம் 2வது தெரு வார்டு என் 27"
            },
            {
                "section_number": "3",
                "section_name": "திருப்பூர் (மா), முருங்கப்பாளையம் வார்டு எண்- 27"
            },
            {
                "section_number": "4",
                "section_name": "திருப்பூர் (மா), முருங்கப்பாளையம் தெற்கு இட்டேரி வார்டு எண்-27"
            }
            ],
            "polling_station_number": null,
            "polling_station_name": "ஈதான் கார்டன் நர்சரி அன்டு பிரைமரி பள்ளி, முருங்கப்பாளையம் இட்டேரி ரோடு",
            "polling_station_address": "திருப்பூர்-641602, வடக்கு பார்த்த கிழக்கு பகுதி மாடியிலிருந்து 1வது அறை (யு.கே.ஜி-இ)",
            "polling_station_type": null,
            "auxiliary_polling_station_count": 0
        },
        "detailed_elector_summary": {
            "serial_number_range": {
            "start": null,
            "end": null
            },
            "mother_roll": {
            "male": 697,
            "female": 702,
            "third_gender": 0,
            "total": 1399
            },
            "additions": {
            "male": 5,
            "female": 9,
            "third_gender": 0,
            "total": 14
            },
            "deletions": {
            "male": 8,
            "female": 5,
            "third_gender": 0,
            "total": 13
            },
            "gender_modification_difference": {
            "male": 0,
            "female": 0,
            "third_gender": 0,
            "total": 0
            },
            "net_total": {
            "male": 694,
            "female": 706,
            "third_gender": 0,
            "total": 1400
            }
        },
        "modifications_info": {
            "roll_type": null,
            "roll_identification": null,
            "total_modifications": null
        },
        "authority_verification": {
            "designation": null,
            "signature_present": false
        },
        "voters": [],
        "ai_metadata": {
            "provider": "gemini",
            "base_url": "https://api.groq.com/openai/v1/",
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "pricing": {
            "currency": "USD",
            "input_cost_per_1m_tokens": 0.2,
            "output_cost_per_1m_tokens": 0.6
            },
            "usage": {
            "prompt_tokens": 5518,
            "completion_tokens": 997,
            "total_tokens": 6515
            },
            "estimated_cost": {
            "currency": "USD",
            "value": 0.0017018000000000003
            }
        }
        }
    ```

3. skip the page which doesn't have the voters information. (openCV will be used for this purpose)

4. Cropping the voter list from the pages where the voters are listed.
each page will have 3 columns and each column will have around 10 rows. so the cropped image will be divided into 3 equal parts vertically and each part will be further divided into 10 equal parts horizontally. (openCV will be used for this purpose, by shape detection and contour detection.)
    3 x 10 = 30 images for each page.

5. Each cropped part will be processed using OCR to extract text. (Tesseract OCR will be used for this purpose)
6. The extracted text will be cleaned and formatted into a structured format (like CSV or JSON) for easy access and analysis.
5. The extracted text will be structured into a tabular format with the following columns

7. Finally, the structured data will be saved to a json file and to the database.