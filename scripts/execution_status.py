import os
from scripts.signature_blocks import SignatureBlockModel
from typing import List, Any, Tuple


def flatten_list(xss: List[List[Any]]) -> List[Any]:
    return [x for xs in xss for x in xs]


def agreement_status(labels: List[str]) -> str:
    if labels:
        if len(set(labels)) > 1:
            return "Partially Executed"
        elif list(set(labels))[0] == "SIGNED_BLOCK":
            return "Fully Executed"
        elif list(set(labels))[0] == "UNSIGNED_BLOCK":
            return "Unsigned"
    else:
        return "Unknown"


def execution_status(
    images: List[Any], show: bool = False
) -> (int, str, List[Any], List[Any]):
    if isinstance(images, list):
        labels = []
        boxes = []
        crops = []
        for page in images:
            model = SignatureBlockModel(page)
            if model.predictions[0]["boxes"].shape[0] > 0:
                page_labels = model._get_labels_names()
                labels.append(page_labels)
                boxes.extend(model.get_box_crops())
                crops.extend(model.get_boxes())
                if show:
                    boxes = model.show_boxes()
        # page.close()
        num_sig_pages = len(labels)
        execution_status = agreement_status(flatten_list(labels))
        return num_sig_pages, execution_status, boxes, crops
    else:
        return None, None, None, None


if __name__ == "__main__":
    from gabriel.parsers.pdf_parser import ParsePDF

    filepath = "/Users/jordandavis/GitHub/gabriel/gabriel/datasets/MASTER_REVIEWED/SIGNATURE_PAGE/1a90afa457f328fc7f560d9b49af7b8f.pdf"
    image = list(ParsePDF(filepath).yield_image())[0]

    num_sig_pages, status, boxes, crops = execution_status(image)
    print(f"Num Sig Pages: {num_sig_pages}")
    print(f"Status: {status}")
    print(f"Boxes: {boxes}")
    print(f"Crops: {crops}")
