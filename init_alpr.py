from fast_alpr import ALPR

# You can also initialize the ALPR with custom plate detection and OCR models.
alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)

alpr_s = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end",
    ocr_model="cct-s-v1-global-model",
)
