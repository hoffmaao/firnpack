# WAIS Divide (Antarctica) staged datasets

This folder is created by `download_summit_wais_data.py`.

## Structure
- `sources/`  symlinks (or copies) to downloaded source datasets
- `manual/`   drop-in folder for any files you had to fetch manually
- `processed/` (optional) your own processed/cleaned tables for inversion

If a download requires credentials/CAPTCHA, you can place files in `manual/` and keep
your inversion scripts stable by pointing at this folder.
