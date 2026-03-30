# ChaosScore

Docker setup (recommended on Windows for `cellxgene-census` compatibility):

1. Build the image:
   `docker compose build`
2. Start an interactive shell with the whole repo mounted:
   `docker compose run --rm chaosscore`
3. Inside the container, run your script:
   `python src/download_lung_reference_data.py --help`

Notes:
- Your local project folder is mounted at `/workspace` inside the container.
- Any edits on Windows are immediately visible inside the container.
