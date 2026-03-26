import random, csv, os
from datetime import datetime
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

TEMPLATES = {
    "billing": {
        "negative": [
            "I was charged twice for my subscription this month. Order #%ORDER%. Please fix this immediately.",
            "My invoice shows $%AMT% but I was told it would be $%LAMT%. This is unacceptable.",
            "Why am I being charged for a service I cancelled %DAYS% days ago? I want a full refund.",
            "I noticed an unauthorized charge of $%AMT% on my account. I did NOT approve this.",
            "Your billing system is broken. I have been overcharged for the %NTH% time. Fix this or I am leaving.",
            "I keep getting charged even after downgrading my plan. This is borderline fraudulent.",
            "The promo code %PROMO% was supposed to give me %DISC% percent off but I was charged full price.",
            "I have been waiting %DAYS% days for my refund of $%AMT%. Where is my money?",
        ],
        "positive": [
            "Just wanted to say thanks for sorting out my billing issue so quickly. Great service!",
            "The refund for order #%ORDER% came through perfectly. Appreciate the fast turnaround.",
            "Your new billing dashboard is really easy to use. Love the invoice download feature.",
        ],
        "neutral": [
            "Can you explain the charges on my latest invoice? I see a line item I do not recognize.",
            "I would like to switch from monthly to annual billing. What are my options?",
            "When will my next invoice be generated? I need it for expense reporting.",
        ],
    },
    "shipping": {
        "negative": [
            "My order #%ORDER% has not arrived and it has been %DAYS% days. Tracking shows no updates.",
            "Package was delivered to the wrong address. This is the %NTH% time this has happened.",
            "My item arrived completely damaged. The box was crushed and the product inside is broken.",
            "I paid for express shipping but it is taking longer than standard delivery. Ridiculous.",
            "The tracking number %TRACK% does not work. No idea where my package is.",
            "Received someone elses order instead of mine. How does this even happen?",
            "My order was marked as delivered but I never received it.",
        ],
        "positive": [
            "Package arrived a day early! Everything was perfectly wrapped. Impressed with the shipping.",
            "Love that you now offer same-day delivery in my area. Just ordered and it arrived in 3 hours!",
            "Thank you for the free shipping upgrade on my last order. That was a nice surprise.",
        ],
        "neutral": [
            "What shipping options are available for international orders to %COUNTRY%?",
            "Can I change the delivery address for order #%ORDER%?",
            "Is there a way to schedule delivery for a specific date?",
        ],
    },
    "product_defect": {
        "negative": [
            "The screen on my %PRODUCT% stopped working after just %DAYS% days. Terrible quality.",
            "Battery drains from 100 to 0 in %HOURS% hours. Completely unusable.",
            "The %COMPONENT% broke on the first use. Clearly a manufacturing defect.",
            "My %PRODUCT% makes a loud grinding noise whenever I turn it on.",
            "The paint is already chipping off after %WEEKS% weeks. What kind of quality control is this?",
            "Software keeps crashing every time I try to use the %FEATURE% function. Totally unreliable.",
            "Bought this %PRODUCT% %MONTHS% months ago and it is already falling apart. Never buying again.",
        ],
        "positive": [
            "The new %PRODUCT% update fixed all the issues I was having. Works perfectly now!",
            "Build quality on the latest model is noticeably better than previous versions. Good job!",
            "Been using my %PRODUCT% daily for %MONTHS% months and it still works like new.",
        ],
        "neutral": [
            "Is the %COMPONENT% issue on the %PRODUCT% covered under warranty?",
            "What is the expected lifespan of the %PRODUCT% battery?",
            "Are there any known issues with the latest %PRODUCT% firmware update?",
        ],
    },
    "cancellation": {
        "negative": [
            "I want to cancel my subscription effective immediately. Your service has been terrible.",
            "Please stop charging my card. I have tried to cancel %ATTEMPTS% times through your website.",
            "Cancel my account NOW. I have been trying for %DAYS% days and nobody is helping me.",
            "Your cancellation process is intentionally difficult. Just let me cancel.",
            "I am done with this service. Too many issues, too little support. Cancel everything.",
            "If you cannot resolve this billing issue today, I want a full cancellation and refund.",
        ],
        "positive": [
            "I need to cancel for now but your team has been great. I will be back when budget allows.",
            "Cancelling because I am switching workflows, not because of any issues. Thanks for everything!",
        ],
        "neutral": [
            "What happens to my data if I cancel? Can I export it first?",
            "Is there a way to pause my subscription instead of cancelling?",
            "What is the cancellation policy for annual plans?",
        ],
    },
    "feature_request": {
        "negative": [
            "I cannot believe you still do not have %FEATURE%. Every competitor has this.",
            "Without %FEATURE%, this product is basically useless for my workflow.",
            "Been requesting %FEATURE% for over a year now. You do not listen to customers.",
        ],
        "positive": [
            "It would be great if you could add %FEATURE%. Would make the product even better!",
            "Love the product! One suggestion: adding %FEATURE% would be amazing for power users.",
            "The new update is fantastic. Any plans to add %FEATURE%? That would make it perfect.",
        ],
        "neutral": [
            "Is %FEATURE% on the roadmap? I would find it really useful for my work.",
            "Can you add %FEATURE%? I have seen similar tools offer this.",
            "Would it be possible to customize the %COMPONENT% settings?",
        ],
    },
    "account_access": {
        "negative": [
            "I cannot log in to my account. Reset my password %ATTEMPTS% times, still does not work.",
            "My account got locked for no reason. I need access RIGHT NOW.",
            "Two-factor auth is not sending codes to my phone. I am completely locked out.",
            "Someone hacked my account and changed the email. I need this resolved urgently.",
            "Your login page keeps throwing errors. Cannot access my account in %DAYS% days.",
        ],
        "positive": [
            "The new login process with biometrics is so much smoother. Thanks for the upgrade!",
            "Support helped me recover my account within minutes. Excellent response time.",
        ],
        "neutral": [
            "How do I enable two-factor authentication on my account?",
            "Can I change the email address associated with my account?",
            "What are the password requirements?",
        ],
    },
}

PRODUCTS = ["laptop","tablet","phone","headphones","speaker","smartwatch","router","camera"]
COMPONENTS = ["screen","battery","keyboard","touchpad","hinge","charging port","speaker","microphone"]
FEATURES = ["dark mode","bulk export","API access","offline mode","team sharing","custom reports","mobile app","calendar integration","Slack integration","auto-save","version history"]
COUNTRIES = ["Indonesia","Japan","Germany","Brazil","UK","Australia","Canada","India"]
PROMO_CODES = ["SAVE20","WELCOME10","ANNUAL30","FLASH50","LOYALTY15","SUMMER25"]
NOISE_TYPOS = {"the":["teh","hte"],"this":["tihs","thsi"],"please":["pleas","pls","plz"],"because":["becuase","cuz","bc"],"received":["recieved","recived"],"working":["wrking","workng"],"problem":["prolbem","problm"]}


def add_noise(text, prob=0.15):
    words = text.split()
    out = []
    for w in words:
        lw = w.lower().strip(".,!?")
        if lw in NOISE_TYPOS and random.random() < prob:
            out.append(w.replace(lw, random.choice(NOISE_TYPOS[lw])))
        else:
            out.append(w)
    result = " ".join(out)
    if random.random() < 0.05:
        result = result.upper()
    if random.random() < 0.1:
        result = random.choice(["hey, ","hi, ","um, ","so, ","look, ","ok so ","honestly, "]) + result[0].lower() + result[1:]
    return result


def fill_template(t):
    r = {"%ORDER%":str(random.randint(100000,999999)),"%AMT%":str(random.randint(10,500)),
         "%LAMT%":str(random.randint(5,100)),"%DAYS%":str(random.randint(2,30)),
         "%HOURS%":str(random.randint(1,8)),"%WEEKS%":str(random.randint(1,12)),
         "%MONTHS%":str(random.randint(1,18)),"%ATTEMPTS%":str(random.randint(2,7)),
         "%NTH%":random.choice(["2nd","3rd","4th","5th"]),
         "%TRACK%":fake.bothify("??########"),"%PRODUCT%":random.choice(PRODUCTS),
         "%COMPONENT%":random.choice(COMPONENTS),"%FEATURE%":random.choice(FEATURES),
         "%COUNTRY%":random.choice(COUNTRIES),"%PROMO%":random.choice(PROMO_CODES),
         "%DISC%":str(random.choice([10,15,20,25,30,50]))}
    for k,v in r.items():
        t = t.replace(k,v)
    return t


def generate_customer_pool(n=500):
    return [{"customer_id":f"CUST-{i+1:05d}","name":fake.name(),"email":fake.email(),
             "signup_date":fake.date_between(start_date="-2y",end_date="-30d"),
             "plan":random.choice(["free","basic","pro","enterprise"]),
             "lifetime_value":round(random.uniform(0,5000),2)} for i in range(n)]


def generate_tickets(n_tickets=10000, n_customers=500):
    customers = generate_customer_pool(n_customers)
    start = datetime(2024,1,1)
    end = datetime(2026,3,1)
    cat_w = {"billing":0.25,"shipping":0.20,"product_defect":0.20,"cancellation":0.10,"feature_request":0.10,"account_access":0.15}
    sent_w = {"billing":{"negative":0.60,"neutral":0.25,"positive":0.15},
              "shipping":{"negative":0.55,"neutral":0.25,"positive":0.20},
              "product_defect":{"negative":0.65,"neutral":0.20,"positive":0.15},
              "cancellation":{"negative":0.70,"neutral":0.20,"positive":0.10},
              "feature_request":{"negative":0.30,"neutral":0.40,"positive":0.30},
              "account_access":{"negative":0.55,"neutral":0.30,"positive":0.15}}
    cats = list(cat_w.keys())
    tickets = []
    for i in range(n_tickets):
        cust = random.choice(customers[:50]) if random.random()<0.3 else random.choice(customers)
        cat = random.choices(cats, weights=list(cat_w.values()), k=1)[0]
        sw = sent_w[cat]
        sent = random.choices(list(sw.keys()), weights=list(sw.values()), k=1)[0]
        text = fill_template(random.choice(TEMPLATES[cat][sent]))
        if random.random() < 0.30:
            text = add_noise(text)
        created = fake.date_time_between(start_date=start, end_date=end)
        dtr = random.randint(0,2) if sent=="positive" else random.randint(1,14) if sent=="neutral" else random.randint(1,30)
        status = random.choices(["resolved","pending","escalated"],
                                weights=[0.65,0.20,0.15] if cat!="cancellation" else [0.40,0.30,0.30], k=1)[0]
        tickets.append({"ticket_id":f"TKT-{i+1:06d}","customer_id":cust["customer_id"],
            "customer_name":cust["name"],"customer_plan":cust["plan"],
            "created_date":created.strftime("%Y-%m-%d %H:%M:%S"),"category":cat,
            "sentiment_label":sent,"text":text,"resolution_status":status,
            "days_to_resolve":dtr,"is_repeat_contact":random.random()<0.25,
            "priority":"high" if sent=="negative" and cat in ("cancellation","account_access") else "medium" if sent=="negative" else "low"})
    return tickets


def save_tickets(tickets, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tickets[0].keys())
        w.writeheader()
        w.writerows(tickets)
    print(f"[OK] Saved {len(tickets):,} tickets to {path}")


if __name__ == "__main__":
    print("Generating synthetic support tickets...")
    tickets = generate_tickets(10000, 500)
    save_tickets(tickets, "data/synthetic/support_tickets.csv")
    from collections import Counter
    cats = Counter(t["category"] for t in tickets)
    sents = Counter(t["sentiment_label"] for t in tickets)
    print(f"  Categories: {dict(cats.most_common())}")
    print(f"  Sentiments: {dict(sents.most_common())}")
