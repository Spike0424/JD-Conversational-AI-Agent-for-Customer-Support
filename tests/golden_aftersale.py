"""黄金测试集：售后场景 · iPhone 17 系列。

基于 10 个售后客服话术场景，覆盖退货、退款、换货、质量问题、商品不符等。
所有问题来自 user_002（已签收 15 个订单），scene_classifier 自然判定为 aftersale。

调用：
    from tests.golden_aftersale import DATASET
    from app.evaluation.runner import EvalRunner
    report = EvalRunner().run(DATASET, dataset_name="golden_aftersale")
"""

from app.evaluation.schemas import EvalQuery

# user_002 在 seed_orders.py 中有 15 条已签收订单
# scene_classifier 通过 arr_time NOT NULL → all_signed=True → aftersale
_AFTERSALE_USER = "user_002"

DATASET: list[EvalQuery] = [
    # ═══════════════════════════════════════════════════════════════════
    # 1. 退货政策 — 七天无理由退货
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_01_7day_return",
        question="iPhone 17 Pro 昨天刚收到，还没拆封，可以七天无理由退货吗？怎么操作？",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["7天无理由退货", "退货流程", "退货政策"],
        noise_contexts=["京东物流次日达", "PLUS会员免运费"],
        ground_truth_answer="支持7天无理由退货。在京东「我的订单」中申请退货，保持商品原包装和完好状态，不影响二次销售即可。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 2. 纠纷处理 — 投诉/平台介入
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_02_dispute",
        question="iPhone 17 用了三天发现屏幕有划痕，客服说是我自己弄的不给退，我要投诉！申请京东介入！",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["纠纷处理", "投诉", "平台介入"],
        noise_contexts=["京东配送范围覆盖全国", "预计3-5个工作日送达"],
        ground_truth_answer="建议先提供问题照片，客服会优先协商解决。如果无法达成一致，可通过京东客服介入处理，京东会审核并保护您的权益。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 3. 评价引导 — 引导好评
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_03_review_guide",
        question="iPhone 17 Pro Max 用着挺好的，但商家打电话让我给好评，合理吗？我不太想评。",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["评价引导", "好评"],
        noise_contexts=["京东PLUS会员开卡福利", "满199减30优惠券"],
        ground_truth_answer="可以请求买家留下评价，也可以提供优惠券感谢好评，但禁止好评返现，禁止威胁或骚扰买家修改评价。您可以自愿选择是否评价。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 4. 退款处理 — 退款进度
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_04_refund_status",
        question="iPhone 17 的退货你们签收三天了，退款怎么还没到账？什么时候退？",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["退款处理", "退款流程", "退款进度"],
        noise_contexts=["京东快递上门取件", "当日达服务"],
        ground_truth_answer="退货商品经京东或仓库验收后，退款将原路退回您的支付账户。通常需要几个工作日，若超时未到账请联系客服协助查询。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 5. 换货处理 — 换货流程
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_05_exchange",
        question="iPhone 17 Pro 256G 买错了，我想换 512G 的，怎么换？",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["换货处理", "换货流程"],
        noise_contexts=["京东自营正品保障", "品牌官方授权"],
        ground_truth_answer="在京东「我的订单」中申请退货，退货审核通过后重新下单购买需要的商品。也可以详细说明问题，客服会推荐合适的替换方案。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 6. 质量问题 — 商品质量问题
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_06_quality",
        question="iPhone 17 的屏幕有个亮点，而且充电到80%就充不进去了，这是质量问题吧？怎么办？",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["质量问题处理", "质量问题"],
        noise_contexts=["iPhone 17 搭载全新A19芯片", "超视网膜XDR显示屏"],
        ground_truth_answer="请提供问题描述和清晰照片或视频。可以选择的方案：换货、部分退款或全额退款。根据京东政策，质量问题无需承担运费。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 7. 商品不符 — 收到的商品与描述不符
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_07_mismatch",
        question="我买的 iPhone 17 Pro 页面写的是钛金属边框，收到的怎么是不锈钢的？这跟描述不一样啊！",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["商品不符处理", "商品不符", "描述不符"],
        noise_contexts=["iPhone 17 支持5G双卡双待", "4800万像素主摄"],
        ground_truth_answer="请提供收到的商品照片，会尽快安排解决方案：发送正确商品或退款处理。这是我们的失误，会第一时间处理。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 8. 退货地址 — 退货地址咨询
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_08_return_address",
        question="iPhone 17 退货地址是什么？我申请好了，但不知道寄到哪里。",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["退货地址提供", "退货地址"],
        noise_contexts=["京东仓储覆盖全国", "次日达承诺"],
        ground_truth_answer="退货申请通过后，在退货页面会看到京东提供的退货地址或二维码。可以选择填写地址寄回，或使用二维码到指定地点寄回（京东快递上门取件）。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 9. 部分退款 — 部分退款协商
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_09_partial_refund",
        question="iPhone 17 包装盒有点压坏了但手机没事，退货太麻烦，能退点钱吗？",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["部分退款处理", "部分退款"],
        noise_contexts=["京东快递24小时客服热线", "电子发票可在线申请"],
        ground_truth_answer="可以提供部分退款，您可保留商品无需退货。如需此方案，请确认金额后提交部分退款申请。",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 10. 满意度回访 — 购买后回访
    # ═══════════════════════════════════════════════════════════════════
    EvalQuery(
        query_id="aftersale_10_followup",
        question="你们客服打电话回访问我 iPhone 17 Pro Max 用得怎么样，我觉得不错，但想问一下有没有什么保养建议？",
        user_id=_AFTERSALE_USER,
        expected_scene="aftersale",
        relevant_source_patterns=["客户满意度回访", "回访"],
        noise_contexts=["下单后30分钟内极速发货", "京东物流实时的跟踪信息"],
        ground_truth_answer="会跟进使用情况，如有任何问题或疑虑可随时通过京东客服联系我们。可以建议定期清洁、使用原装充电器、避免过度充放电等保养方法。",
    ),
]