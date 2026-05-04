def detect_intent(question: str) -> str:
    text = question.lower()

    if any(keyword in text for keyword in ["订单", "物流", "发货", "order", "shipment"]):
        return "order_query"
    if any(keyword in text for keyword in ["保修", "维修", "warranty", "repair"]):
        return "warranty_service"
    if any(keyword in text for keyword in ["退货", "换货", "售后", "refund", "return"]):
        return "after_sale_service"
    if any(keyword in text for keyword in ["推荐", "对比", "参数", "配置", "recommend", "compare"]):
        return "product_consulting"
    return "general_qa"
