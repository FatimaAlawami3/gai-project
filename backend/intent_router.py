import re

from prompts import detect_language


ROAD_SAFETY_PATTERNS = {
    "en": [
        r"\btraffic\b",
        r"\broad\b",
        r"\bdriv(e|er|ing)\b",
        r"\bcar\b",
        r"\bautomobile\b",
        r"\bvehicle\b",
        r"\blicen[sc]e\b",
        r"\bregistration\b",
        r"\binspection\b",
        r"\bpermit\b",
        r"\bpermission\b",
        r"\bconsequence(s)?\b",
        r"\bviolation\b",
        r"\bfine\b",
        r"\bpenalt(y|ies)\b",
        r"\bmodif(y|ying|ication|ications)\b",
        r"\balter(s|ed|ing|ation|ations)?\b",
        r"\bchange(s|d|ing)?\b.*\bcolo(u)?r\b",
        r"\bcolo(u)?r\b.*\bchange(s|d|ing)?\b",
        r"\bvehicle'?s?\s+colo(u)?r\b",
        r"\bcar'?s?\s+colo(u)?r\b",
        r"\brepair\s+shop\b",
        r"\bworkshop\b",
        r"\bspeed\b",
        r"\bstopping distance\b",
        r"\bbraking distance\b",
        r"\breaction distance\b",
        r"\bsign\b",
        r"\bsignal\b",
        r"\btraffic light(s)?\b",
        r"\bred light\b",
        r"\bgreen light\b",
        r"\bamber light\b",
        r"\byellow light\b",
        r"\bstop line\b",
        r"\broundabout\b",
        r"\bintersection\b",
        r"\bparking\b",
        r"\bstopping\b",
        r"\bpedestrian\b",
        r"\bcrosswalk\b",
        r"\baccident\b",
        r"\bdamage\b",
        r"\bdamaged\b",
        r"\binjur(y|ies)\b",
        r"\bnajm\b",
        r"\bred crescent\b",
        r"\blane\b",
        r"\bovertak(e|ing)\b",
        r"\bheadlight\b",
        r"\bseat\s*belt\b",
        r"\bright[- ]of[- ]way\b",
        r"\byield\b",
        r"\bpriority\b",
        r"\bmoroor\b",
        r"\bsaudi\s+traffic\b",
        r"\bphone\b",
        r"\bmobile\b",
        r"\bcell\s*phone\b",
        r"\bhands[- ]free\b",
        r"\bdevice\b",
        r"\bdistract(ed|ion)?\b",
    ],
    "ar": [
        r"مرور",
        r"طريق",
        r"طرق",
        r"قياد",
        r"سائق",
        r"مركب",
        r"سيار",
        r"رخص",
        r"استمار",
        r"فحص",
        r"تصريح",
        r"إذن",
        r"اذن",
        r"عواقب",
        r"نتائج",
        r"مخالف",
        r"غرام",
        r"عقوب",
        r"تعديل",
        r"تغيير",
        r"لون",
        r"طلاء",
        r"صبغ",
        r"ورشة",
        r"إصلاح",
        r"اصلاح",
        r"سرع",
        r"مسافة التوقف",
        r"مسافة الكبح",
        r"مسافة الفرملة",
        r"مسافة رد الفعل",
        r"إشار",
        r"اشار",
        r"علام",
        r"لوح",
        r"إشارة ضوئية",
        r"اشارة ضوئية",
        r"إشارة المرور",
        r"اشارة المرور",
        r"ضوء أحمر",
        r"ضوء اخضر",
        r"ضوء أصفر",
        r"دوار",
        r"تقاطع",
        r"وقوف",
        r"مواقف",
        r"مشا",
        r"حادث",
        r"حوادث",
        r"أضرار",
        r"اضرار",
        r"تلف",
        r"إصابة",
        r"اصابة",
        r"إصابات",
        r"اصابات",
        r"نجم",
        r"الهلال الأحمر",
        r"مسار",
        r"تجاوز",
        r"أولوية",
        r"اولوي",
        r"مصابيح",
        r"حزام",
        r"هاتف",
        r"جوال",
        r"الهاتف",
        r"الجوال",
        r"بدون يد",
        r"بدون استخدام اليد",
        r"تشتيت",
        r"انشغال",
    ],
}


GREETING_PATTERNS = {
    "en": [
        r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\s*[!.?]*\s*$",
        r"\bhow are you\b",
    ],
    "ar": [
        r"^\s*(مرحبا|مرحباً|اهلا|أهلا|السلام عليكم|هلا)\s*[!.؟]*\s*$",
        r"كيف حالك",
    ],
}


THANKS_PATTERNS = {
    "en": [r"\bthank(s| you)\b", r"\bappreciate it\b"],
    "ar": [r"شكرا", r"شكراً", r"يعطيك العافية"],
}


CAPABILITY_PATTERNS = {
    "en": [
        r"\bwhat can you do\b",
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bhelp\b",
        r"\bhow do I use\b",
    ],
    "ar": [
        r"ماذا تستطيع",
        r"وش تقدر",
        r"من أنت",
        r"من انت",
        r"ساعد",
        r"كيف أستخدم",
        r"كيف استخدم",
    ],
}


PROJECT_INFO_PATTERNS = {
    "en": [
        r"\bproject\b",
        r"\bteam\b",
        r"\bmember(s)?\b",
        r"\bworked on\b",
        r"\bwho (built|made|created|developed)\b",
        r"\bdeveloper(s)?\b",
        r"\bstudent(s)?\b",
        r"\bsenior ai\b",
        r"\bcourse\b",
        r"\bgenerative ai course\b",
        r"\buniversity\b",
        r"\biau\b",
        r"\bsupervisor\b",
        r"\bsupervised\b",
        r"\bdr\.?\s*mustafa\b",
        r"\bmustafa\s+m\.?\s+youldash\b",
        r"\bhaya\b",
        r"\braneem\b",
        r"\banfal\b",
        r"\bnorah\b",
        r"\bfatimah\b",
        r"\bfai\b",
    ],
    "ar": [
        r"مشروع",
        r"الفريق",
        r"أعضاء",
        r"اعضاء",
        r"من عمل",
        r"مين عمل",
        r"من طور",
        r"مين طور",
        r"الطالبات",
        r"طلاب",
        r"طالبات",
        r"ذكاء اصطناعي",
        r"الذكاء الاصطناعي",
        r"مقرر",
        r"جامعة",
        r"iau",
        r"إشراف",
        r"اشراف",
        r"المشرف",
        r"مصطفى",
        r"يولداش",
        r"haya",
        r"raneem",
        r"anfal",
        r"norah",
        r"fatimah",
        r"fai",
    ],
}


PROJECT_DETAIL_PATTERNS = {
    "project_name": {
        "en": [r"\b(project )?name\b", r"\bwhat is it called\b"],
        "ar": [r"اسم", r"وش اسمه", r"ما اسمه"],
    },
    "team": {
        "en": [
            r"\bteam\b",
            r"\bmember(s)?\b",
            r"\bbehind\b",
            r"\bworked on\b",
            r"\bwho (built|made|created|developed)\b",
            r"\bdeveloper(s)?\b",
            r"\bstudent(s)?\b",
            r"\bhaya\b",
            r"\braneem\b",
            r"\banfal\b",
            r"\bnorah\b",
            r"\bfatimah\b",
            r"\bfai\b",
        ],
        "ar": [
            r"الفريق",
            r"أعضاء",
            r"اعضاء",
            r"وراء",
            r"خلف",
            r"من عمل",
            r"مين عمل",
            r"من طور",
            r"مين طور",
            r"الطالبات",
            r"طلاب",
            r"طالبات",
        ],
    },
    "supervisor": {
        "en": [
            r"\bsupervisor\b",
            r"\bsupervised\b",
            r"\bdr\.?\s*mustafa\b",
            r"\bmustafa\s+m\.?\s*youldash\b",
        ],
        "ar": [r"إشراف", r"اشراف", r"المشرف", r"مصطفى", r"يولداش"],
    },
    "course": {
        "en": [r"\bcourse\b", r"\bgenerative ai\b", r"\biau\b", r"\buniversity\b"],
        "ar": [r"مقرر", r"جامعة", r"iau", r"ذكاء اصطناعي", r"الذكاء الاصطناعي"],
    },
}


ANSWER_INTENT_PATTERNS = {
    "definition": {
        "en": [
            r"\bwhat is\b",
            r"\bwhat does\b.*\bmean\b",
            r"\bdefine\b",
            r"\bdefinition\b",
            r"\bmeaning\b",
        ],
        "ar": [r"ما معنى", r"ما هو", r"ما هي", r"عرّف", r"عرف", r"تعريف", r"معنى"],
    },
    "penalty_consequence": {
        "en": [
            r"\bpenalt(y|ies)\b",
            r"\bfine(s)?\b",
            r"\bpunishment\b",
            r"\bconsequence(s)?\b",
            r"\bwhat happens\b",
            r"\bif .*violate\b",
            r"\bviolation\b",
        ],
        "ar": [r"عقوب", r"غرام", r"مخالف", r"عواقب", r"نتائج", r"ماذا يحدث", r"وش يصير"],
    },
    "permission_rule": {
        "en": [
            r"\bis .*allowed\b",
            r"\bcan i\b",
            r"\bcan a\b",
            r"\bmay i\b",
            r"\bpermitted\b",
            r"\blegal\b",
            r"\brule(s)?\b",
            r"\brequirement(s)?\b",
        ],
        "ar": [r"هل يسمح", r"مسموح", r"ممنوع", r"نظام", r"قاعدة", r"يجوز", r"قانوني"],
    },
    "procedure": {
        "en": [
            r"\bwhat should\b",
            r"\bwhat do i do\b",
            r"\bhow do i\b",
            r"\bsteps?\b",
            r"\bprocedure\b",
            r"\bwhen approaching\b",
            r"\bafter\b.*\baccident\b",
        ],
        "ar": [r"ماذا يجب", r"ما الذي يجب", r"كيف", r"خطوات", r"إجراءات", r"اجراءات", r"عند"],
    },
    "comparison": {
        "en": [
            r"\bdifference between\b",
            r"\bcompare\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bwhich is\b.*\bsafer\b",
        ],
        "ar": [r"الفرق بين", r"قارن", r"مقارنة", r"أيهما", r"ايهما"],
    },
}


CLARIFICATION_PATTERNS = {
    "en": [
        r"^\s*what happens after an accident\s*[?.!]?\s*$",
        r"^\s*what should i do after an accident\s*[?.!]?\s*$",
        r"^\s*tell me about accidents\s*[?.!]?\s*$",
    ],
    "ar": [
        r"ماذا يحدث بعد حادث",
        r"ماذا أفعل بعد حادث",
        r"ماذا افعل بعد حادث",
        r"حدثني عن الحوادث",
    ],
}


FOLLOWUP_REFERENCE_PATTERNS = {
    "en": [
        r"\bwhat about\b",
        r"\bwhat happens\b",
        r"\bhow about\b",
        r"\bthen\b",
        r"\bthat\b",
        r"\bthis\b",
        r"\bit\b",
        r"\bdo that\b",
        r"\bwhat if\b",
        r"\bonly\b",
        r"\bdamage\b",
        r"\bdamaged\b",
        r"\binjur(y|ies)\b",
    ],
    "ar": [
        r"ماذا عن",
        r"وماذا",
        r"ماذا يحدث",
        r"طيب",
        r"لو",
        r"ولو",
        r"إذا",
        r"اذا",
        r"يعني",
        r"وش",
        r"ايش",
        r"إيش",
        r"ذلك",
        r"هذا",
        r"هذي",
        r"عنها",
        r"ماذا لو",
        r"فقط",
        r"أضرار",
        r"اضرار",
        r"إصابات",
        r"اصابات",
        r"المتسبب",
        r"المتسببة",
        r"الحق علي",
        r"الحق عليّ",
        r"الخطأ علي",
        r"الخطأ عليّ",
        r"الخطأ مني",
        r"الغلطة مني",
        r"أنا السبب",
        r"انا السبب",
        r"أنا المتسبب",
        r"انا المتسبب",
        r"وش يصير",
        r"ايش يصير",
        r"إيش يصير",
    ],
}


ROAD_TOPIC_PATTERNS = {
    "en": {
        "roundabout": [r"\broundabout\b", r"\btraffic circle\b"],
        "parking": [r"\bparking\b", r"\bstopping\b", r"\bwaiting\b"],
        "accident": [
            r"\baccident\b",
            r"\bcollision\b",
            r"\bdamage\b",
            r"\bdamaged\b",
            r"\binjur(y|ies)\b",
            r"\bnajm\b",
            r"\bred crescent\b",
        ],
        "vehicle_color": [
            r"\bcolo(u)?r\b",
            r"\bpaint\b",
            r"\bmodif(y|ying|ication|ications)\b",
            r"\balter(s|ed|ing|ation|ations)?\b",
            r"\b(car|vehicle|automobile)\b.*\bchange(s|d|ing)?\b",
            r"\bchange(s|d|ing)?\b.*\b(car|vehicle|automobile)\b",
        ],
        "driving_license": [
            r"\bdriving licen[sc]e\b",
            r"\bprivate driving licen[sc]e\b",
            r"\bpublic driving licen[sc]e\b",
            r"\bmotorcycle driving licen[sc]e\b",
            r"\btemporary driving licen[sc]e\b",
            r"\bminimum age\b",
            r"\bat least\b.*\byears?\b",
            r"\bobtain(ing)?\b.*\blicen[sc]e\b",
            r"\blicen[sc]e\b.*\b(valid|renew|issue|requirements?)\b",
        ],
        "unlicensed_driver": [r"\bunlicensed\b", r"\blicen[sc]e\b"],
        "speed": [
            r"\bspeed\b",
            r"\bspeeding\b",
            r"\bstopping distance\b",
            r"\bbraking distance\b",
            r"\breaction distance\b",
        ],
        "phone_use": [
            r"\bphone\b",
            r"\bmobile\b",
            r"\bcell\s*phone\b",
            r"\bhands[- ]free\b",
            r"\bdistract(ed|ion)?\b",
            r"\bdevice\b",
        ],
        "road_signs": [
            r"\bsign\b",
            r"\bsignal\b",
            r"\btraffic light(s)?\b",
            r"\bred light\b",
            r"\bgreen light\b",
            r"\bamber light\b",
            r"\byellow light\b",
            r"\bstop line\b",
        ],
        "lane": [r"\blane\b", r"\bovertak(e|ing)\b"],
        "pedestrian": [r"\bpedestrian\b", r"\bcrosswalk\b"],
    },
    "ar": {
        "roundabout": [r"دوار"],
        "parking": [r"وقوف", r"مواقف", r"انتظار"],
        "accident": [
            r"حادث",
            r"حوادث",
            r"تصادم",
            r"صدم",
            r"أضرار",
            r"اضرار",
            r"إصابة",
            r"اصابة",
            r"نجم",
        ],
        "vehicle_color": [
            r"لون",
            r"طلاء",
            r"صبغ",
            r"(?:سيار|مركب).*(?:تعديل|تغيير|شكل)",
            r"(?:تعديل|تغيير|شكل).*(?:سيار|مركب)",
        ],
        "driving_license": [
            r"رخصة قيادة",
            r"رخصه قياده",
            r"رخصة",
            r"رخصه",
            r"استخراج",
            r"إصدار",
            r"اصدار",
            r"تجديد",
            r"صلاحية",
            r"مدة",
            r"سن",
            r"عمر",
            r"الحد الأدنى",
            r"الادنى",
            r"خاصة",
            r"عمومية",
            r"مؤقتة",
        ],
        "unlicensed_driver": [r"رخص", r"مرخص"],
        "speed": [r"سرع", r"مسافة التوقف", r"مسافة الكبح", r"مسافة الفرملة", r"مسافة رد الفعل"],
        "phone_use": [
            r"هاتف",
            r"جوال",
            r"الهاتف",
            r"الجوال",
            r"بدون يد",
            r"بدون استخدام اليد",
            r"تشتيت",
            r"انشغال",
        ],
        "road_signs": [
            r"إشار",
            r"اشار",
            r"علام",
            r"لوح",
            r"إشارة ضوئية",
            r"اشارة ضوئية",
            r"إشارة المرور",
            r"اشارة المرور",
            r"ضوء أحمر",
            r"ضوء اخضر",
            r"ضوء أصفر",
        ],
        "lane": [r"مسار", r"تجاوز"],
        "pedestrian": [r"مشا", r"عبور"],
    },
}


TEAM_MEMBERS = [
    ("Haya Saad Aldossary", "2220004842"),
    ("Raneem Saif Alqahtani", "2220005353"),
    ("Anfal Salah Bamardouf", "2200003568"),
    ("Norah Mohammed Aldossary", "2220006973"),
    ("Fatimah Dhiyaa Alawami", "2220005142"),
    ("Fai Hadi Alotaibi", "2220002599"),
]


SUGGESTED_QUESTIONS = {
    "en": [
        "What should I do after a traffic accident?",
        "Is it allowed to use a phone while driving?",
        "What happens if I drive without a license?",
        "Can I let someone else drive my car?",
        "Is modifying a car allowed?",
        "What should a driver do when approaching a roundabout?",
        "Who is behind this project?",
    ],
    "ar": [
        "ماذا يجب علي فعله بعد وقوع حادث مروري؟",
        "هل يسمح باستخدام الهاتف أثناء القيادة؟",
        "ماذا يحدث إذا قدت بدون رخصة؟",
        "هل يمكنني السماح لشخص آخر بقيادة سيارتي؟",
        "هل يسمح بتعديل السيارة؟",
        "ما الذي يجب على السائق فعله عند الاقتراب من الدوار؟",
        "من وراء هذا المشروع؟",
    ],
}


def _team_members_text() -> str:
    return "\n".join(f"- {name} - {student_id}" for name, student_id in TEAM_MEMBERS)


def project_info_answer(language: str, detail: str = "overview") -> str:
    members = "\n".join(f"- {name} - {student_id}" for name, student_id in TEAM_MEMBERS)

    if language == "ar":
        if detail == "project_name":
            return "اسم المشروع: دليل (DALIL)."
        if detail == "team":
            return (
                "يقف خلف مشروع دليل (DALIL) فريق من طالبات سنة أخيرة في "
                "تخصص الذكاء الاصطناعي في IAU:\n"
                f"{members}\n\n"
                "المشرف: Dr. Mustafa M. Youldash."
            )
        if detail == "supervisor":
            return "المشرف على مشروع دليل (DALIL): Dr. Mustafa M. Youldash."
        if detail == "course":
            return (
                "تم تنفيذ مشروع دليل (DALIL) لمقرر الذكاء الاصطناعي "
                "التوليدي في جامعة الإمام عبدالرحمن بن فيصل (IAU)."
            )
        return (
            "اسم المشروع: دليل (DALIL).\n\n"
            "دليل هو مرشد سلامة مرورية ذكي يهدف إلى دعم الوعي بالسلامة "
            "المرورية والإجابة عن أسئلة السائقين المتعلقة بأنظمة المرور وإرشادات "
            "القيادة في المملكة العربية السعودية.\n\n"
            "تم تنفيذ المشروع لمقرر الذكاء الاصطناعي التوليدي في جامعة الإمام "
            "عبدالرحمن بن فيصل (IAU).\n\n"
            "أعضاء الفريق، وهن طالبات سنة أخيرة في تخصص الذكاء الاصطناعي في IAU:\n"
            f"{members}\n\n"
            "المشرف على المشروع: Dr. Mustafa M. Youldash."
        )

    if detail == "project_name":
        return "The project name is DALIL."
    if detail == "team":
        return (
            "DALIL was developed by senior AI students at IAU:\n"
            f"{members}\n\n"
            "Supervisor: Dr. Mustafa M. Youldash."
        )
    if detail == "supervisor":
        return "DALIL was supervised by Dr. Mustafa M. Youldash."
    if detail == "course":
        return (
            "DALIL was created for the Generative AI course at "
            "Imam Abdulrahman Bin Faisal University (IAU)."
        )
    return (
        "Project name: DALIL.\n\n"
        "DALIL is a Road Safety Guide AI Chatbot designed to support road safety "
        "awareness and answer drivers' questions about traffic rules and driving "
        "guidance in Saudi Arabia.\n\n"
        "The project was created for the Generative AI course at Imam Abdulrahman "
        "Bin Faisal University (IAU).\n\n"
        "Team members, all senior AI students at IAU:\n"
        f"{members}\n\n"
        "Project supervisor: Dr. Mustafa M. Youldash."
    )


GENERIC_ANSWERS = {
    "greeting": {
        "en": (
            "Hello! I am DALIL. I can help with Saudi road safety questions, including traffic "
            "rules, driving guidance, road signs, violations, parking, accidents, "
            "roundabouts, and safe driving practices."
        ),
        "ar": (
            "مرحباً! أنا دليل. أستطيع مساعدتك في أسئلة السلامة المرورية في السعودية، مثل "
            "أنظمة المرور، إرشادات القيادة، اللوحات والإشارات، المخالفات، الوقوف، "
            "الحوادث، الدوارات، وممارسات القيادة الآمنة."
        ),
    },
    "thanks": {
        "en": "You are welcome. Ask me any Saudi road safety or traffic question when you are ready.",
        "ar": "على الرحب والسعة. اسألني عن أي موضوع متعلق بالسلامة المرورية أو أنظمة المرور في السعودية.",
    },
    "capability": {
        "en": (
            "I am DALIL, a Road Safety Guide AI Chatbot designed to support road "
            "safety awareness and answer drivers' questions about traffic rules and "
            "driving guidance in Saudi Arabia. You can ask me about licenses, "
            "violations, road signs, roundabouts, parking, accidents, speed, lanes, "
            "pedestrians, and safe driving behavior."
        ),
        "ar": (
            "أنا دليل (DALIL)، مرشد سلامة مرورية ذكي يهدف إلى دعم الوعي بالسلامة "
            "المرورية والإجابة عن أسئلة السائقين المتعلقة بأنظمة المرور وإرشادات "
            "القيادة في المملكة العربية السعودية. يمكنك سؤالي عن الرخص، المخالفات، "
            "الإشارات، الدوارات، الوقوف، الحوادث، السرعة، المسارات، المشاة، وسلوك "
            "القيادة الآمن."
        ),
    },
    "general": {
        "en": (
            "I am focused on Saudi road safety and traffic guidance, so I cannot answer "
            "general questions here. Please ask about Saudi traffic law, driving rules, "
            "road signs, violations, parking, accidents, or safe driving."
        ),
        "ar": (
            "أنا مخصص للإجابة عن أسئلة السلامة المرورية وأنظمة المرور في السعودية، "
            "لذلك لا أجيب عن الأسئلة العامة هنا. يمكنك السؤال عن نظام المرور السعودي، "
            "قواعد القيادة، الإشارات، المخالفات، الوقوف، الحوادث، أو القيادة الآمنة."
        ),
    },
}


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def detect_project_detail(text: str, language: str) -> str:
    for detail, localized_patterns in PROJECT_DETAIL_PATTERNS.items():
        if _matches(localized_patterns[language], text):
            return detail
    return "overview"


def history_has_road_safety_context(chat_history: list[dict[str, str]] | None) -> bool:
    if not chat_history:
        return False
    user_messages = [
        item
        for item in chat_history[-6:]
        if item and str(item.get("role", "")).strip().lower() == "user"
    ]
    if not user_messages:
        return False
    recent_text = " ".join(
        str(item.get("content", "")) for item in user_messages[-3:] if item
    )
    language = detect_language(recent_text)
    return _matches(ROAD_SAFETY_PATTERNS[language], recent_text.lower())


def road_topics(text: str, language: str) -> set[str]:
    normalized = " ".join(text.lower().split())
    return {
        topic
        for topic, patterns in ROAD_TOPIC_PATTERNS[language].items()
        if _matches(patterns, normalized)
    }


def history_road_topics(chat_history: list[dict[str, str]] | None) -> set[str]:
    if not chat_history:
        return set()
    recent_user_text = " ".join(
        str(item.get("content", ""))
        for item in chat_history[-6:]
        if str(item.get("role", "")).strip().lower() == "user"
    )
    if not recent_user_text:
        return set()
    return road_topics(recent_user_text, detect_language(recent_user_text))


def looks_like_followup(question: str) -> bool:
    language = detect_language(question)
    text = " ".join(question.strip().lower().split())
    if language == "ar":
        return len(text) <= 80 or _matches(
            [
                r"ماذا عن",
                r"وماذا",
                r"طيب",
                r"هل",
                r"وهل",
                r"عن",
                r"لو",
                r"ولو",
                r"إذا",
                r"اذا",
                r"يعني",
                r"وش",
                r"ايش",
                r"إيش",
                r"المتسبب",
                r"الحق علي",
                r"الخطأ",
                r"الغلطة",
                r"أنا السبب",
                r"انا السبب",
            ],
            text,
        )
    return len(text.split()) <= 8 or _matches(
        [r"\bwhat about\b", r"\band\b", r"\bthen\b", r"\bit\b", r"\bthat\b"], text
    )


def is_directly_related_followup(
    question: str,
    chat_history: list[dict[str, str]] | None,
) -> bool:
    if not looks_like_followup(question) or not history_has_road_safety_context(chat_history):
        return False

    language = detect_language(question)
    normalized = " ".join(question.strip().lower().split())
    current_topics = road_topics(question, language)
    previous_topics = history_road_topics(chat_history)

    if current_topics and previous_topics:
        return bool(current_topics & previous_topics)
    if current_topics and not previous_topics:
        return False

    return _matches(FOLLOWUP_REFERENCE_PATTERNS[language], normalized)


def detect_answer_intent(
    question: str,
    language: str,
    is_contextual_followup: bool = False,
) -> str:
    normalized = " ".join(question.strip().lower().split())
    if is_contextual_followup:
        return "followup"
    if _matches(CLARIFICATION_PATTERNS[language], normalized):
        return "clarification"
    for answer_intent, localized_patterns in ANSWER_INTENT_PATTERNS.items():
        if _matches(localized_patterns[language], normalized):
            return answer_intent
    return "general_road_safety"


def detect_intent(
    question: str, chat_history: list[dict[str, str]] | None = None
) -> dict[str, str | bool]:
    language = detect_language(question)
    normalized = " ".join(question.strip().lower().split())
    is_contextual_followup = is_directly_related_followup(question, chat_history)

    if _matches(PROJECT_INFO_PATTERNS[language], normalized):
        intent = "project_info"
        detail = detect_project_detail(normalized, language)
    elif _matches(GREETING_PATTERNS[language], normalized):
        intent = "greeting"
        detail = "default"
    elif _matches(THANKS_PATTERNS[language], normalized):
        intent = "thanks"
        detail = "default"
    elif _matches(CAPABILITY_PATTERNS[language], normalized):
        intent = "capability"
        detail = "default"
    elif is_contextual_followup:
        return {
            "intent": "road_safety",
            "language": language,
            "use_rag": True,
            "detail": "followup",
            "answer_intent": detect_answer_intent(
                question, language, is_contextual_followup=True
            ),
        }
    elif _matches(ROAD_SAFETY_PATTERNS[language], normalized):
        return {
            "intent": "road_safety",
            "language": language,
            "use_rag": True,
            "detail": detect_answer_intent(question, language),
            "answer_intent": detect_answer_intent(question, language),
        }
    else:
        intent = "general"
        detail = "default"

    return {
        "intent": intent,
        "language": language,
        "use_rag": False,
        "detail": detail,
        "answer_intent": detail if intent == "project_info" else intent,
    }


def generic_answer(intent: str, language: str, detail: str = "default") -> str:
    if intent == "project_info":
        return project_info_answer(language, detail)
    return GENERIC_ANSWERS.get(intent, GENERIC_ANSWERS["general"])[language]


def suggested_questions(language: str) -> list[str]:
    return SUGGESTED_QUESTIONS[language]
