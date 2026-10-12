#!/usr/bin/env bash
# Downloads, verifies (SHA1) and extracts xView2 train / tier3 / test.
# Resumable: re-run the script if the connection drops. Needs ~60 GB free disk.
set -euo pipefail
mkdir -p data && cd data

TRAIN_URL='https://download.xview2.org/train_images_labels_targets.tar.gz?Expires=1790616285&Signature=IkU1HztZILrVy8TIaFMufwL69J3iGkYKkySp9Y5u7POhGne1yW9SbPSiubB016HQlQxq2DVeb~kR7wI3niKvP5iCbs2kAKYQQ97-W4UB8TECLQPKvxRy2pEuEmHX-bsmsVfVlGp48xbG56KpHNzgIkOLSsLDP8g1Ae2DFtkje4VDgsicHEdRvz-5gtHLnxShZATyYzI~riUSyS8sLXFkqk642SsfkUb3cZIhhZyntDHDOX5SQL-0s2rIcrdyj2xZ3ff3pahMMPrD5-bhLZRpHq3~UHnWoCi--3G0~ZL4OyBJVZ4LyBDKIwzpAZ~Y2ytdk91cc~4KyyZEJRISgiQ1Sg__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
TIER3_URL='https://download.xview2.org/tier3.tar.gz?Expires=1790616285&Signature=I3xDsgcsZcGjuGBCIlksP67~TAht2e7DqbfRo3Yij1j8mC7KKCASBFTIPuG-6p5igUL3pz9ITLGtZBOQH2-KIl6AFF2luwrouLY8LPuuy0~lFkCQOOP2oBujsODRt8KHftpiGhSKqXUVqj8Rw~Pw-6JsUBwVUTMqVntlpLb4wjk5BKGoojiY67FUvpCj3c6ZpXIWqcMvmIT89n01UZmjd3g5Hh5t8lD5VV~VCtT19QiMSWSkTUFFCU4UTxK2eCWjIh4af8C3jYUMsLuYdPxdOw0Pbf4MVWyoSZk4kvOmkXnoSkCk0ny-Z56OodzRA1Yv7OxcBOobWeGy0nTEjpsuWw__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'
TEST_URL='https://download.xview2.org/test_images_labels_targets.tar.gz?Expires=1790616285&Signature=KoaqpE1JopGKGDzYIiL--dlNTJMHDZDjfqDpIdnvXyqA14yYyX4JXYUfFrhSGDmRQVk5Av2MQxxHAijpXFOuJPNmSaOwc~Pm0-oYznpxy9363I9VWIGze1ZAkL13g4jB3EpKWO4zXlChr-EVhWbJ-CFVTavM~naTEcWD6kVei1xYQqDTu6LT8f35bkpotOOhVNX4X0UJkloEsYZJ2i-hMeOwMr3Sl-vqFzbvTYMIuCx8pSnXQMNt62Z7EuUNP5m6Z5UPioYdZ4InDKgjO74X2bx6T2MuLjYT5ldB5gj0ICXreOzj2LhoWcYii8gTOXpRLA4L7V7RJmr3IIBuwpF2Sg__&Key-Pair-Id=APKAIKGDJB5C3XUL2DXQ'

fetch () {  # name url sha1
  local f="$1.tar.gz"
  curl -L -C - --retry 5 --retry-delay 5 -o "$f" "$2"
  echo "$3  $f" | sha1sum -c -
  tar -xzf "$f" && echo "extracted $f"
}

fetch train "$TRAIN_URL"  b37a4ef4ee9c909e2b19d046e49d42ee3965714b
fetch test  "$TEST_URL"   86ed3dba2f8d16ceceb75d451005054fefa9616f
fetch tier3 "$TIER3_URL"  5bf6aaf8a71980b633fb4661776a99a200891de5

echo "Done. Expected folders: data/train  data/test  data/tier3 (each with images/ labels/)"
# Optional once verified:  rm *.tar.gz
