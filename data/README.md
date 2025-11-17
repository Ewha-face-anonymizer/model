Data layout

----------

data/input/(원본 이미지), data/output/(결과 이미지), data/logs/에 뭐가 들어가는지 설명합니다. 각 폴더에는 .gitkeep이 있어 빈 디렉터리도 git에 남습니다.

----------
- `input/` : place reference image (`target.jpg`), `test_image.jpg`, or other assets.
- `output/` : pipeline writes mosaicked renders here.
- `logs/` : runtime traces, saved frames, or experiment metadata.

These folders are gitignored by default; copy your own media/models locally.
