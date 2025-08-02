from flask import Flask, request, render_template, request, redirect, url_for, session
import os, fitz, ollama, random, subprocess, re, secrets, requests
from flask_cors import CORS
from flask_mysqldb import MySQL
from flask_session import Session
from datetime import datetime
import json
import os
import MySQLdb.cursors
import MySQLdb.cursors, re, hashlib


app = Flask(__name__)

app.secret_key = secrets.token_hex(16)
app.config["SESSION_TYPE"] = "filesystem"   # 也可 redis, memcached
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'csie'
app.config['MYSQL_DB'] = 'pythonlogin'

mysql=MySQL(app)

Session(app)
CORS(app)  


# -------------------------------------------------------------
#  SoVITS  語音模型設定與 TTS 工具函式
# -------------------------------------------------------------
VOICE_MODELS = {
    "mann": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/mann_e2_s50_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/mann-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\mann.wav",
        "prompt_text": "作者蘆園老師因為當時連載尚未結束，所以一直不想要改編成其他載體，可是製作方不斷釋出他們的誠意，他終究還是開口答應了"
    },
    "mrd": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/mrd_e2_s102_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/mrd-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\mrd.wav",
        "prompt_text": "開始審理案件，被告我看看，在二零二三年接了太多鯊魚廣告，受到觀眾舉發控告。"
    },
    "andy": {
        "sovits": "D:/GPT-SoVITS-v4/SoVITS_weights_v4/andy_e2_s98_l32.pth",
        "gpt": "D:/GPT-SoVITS-v4/GPT_weights_v4/andy-e15.ckpt",
        "ref_audio_path": "D:\GPT-SoVITS-v4\\teacher\\andy.wav",
        "prompt_text": "好玩有趣的創意，用一支手機拍攝起來，開心的分享在社群媒體上面，剛開始都沒有人看"
    }
}
TTS_HOST = "http://127.0.0.1:9880"
_current_sovits = [None]
_current_gpt    = [None]

def _switch_if_needed(cache, new_path, ep, tag):
    if cache[0] == new_path:
        return
    r = requests.get(f"{TTS_HOST}/{ep}", params={"weights_path": new_path}, timeout=120)
    r.raise_for_status()
    cache[0] = new_path

def tts(text: str, model_key: str, out_path="static/output.wav"):
    m = VOICE_MODELS[model_key]
    _switch_if_needed(_current_sovits, m["sovits"], "set_sovits_weights", "SoVITS")
    _switch_if_needed(_current_gpt,    m["gpt"],    "set_gpt_weights",   "GPT")
    params = {
        "text": text, "text_lang": "zh",
        "ref_audio_path": m["ref_audio_path"],
        "prompt_lang": "zh", "prompt_text": m["prompt_text"],
        "text_split_method": "cut1", "batch_size": 2, "sample_steps": 16,
        "media_type": "wav", "streaming_mode": "false"
    }
    r = requests.get(f"{TTS_HOST}/tts", params=params, timeout=240 )
    r.raise_for_status()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)

# -------------------------------------------------------------
#  其餘輔助函式：GPU 檢查、PDF 讀取、成績解析、評分、綜合評語
# -------------------------------------------------------------

def check_gpu():
    try:
        out = subprocess.check_output("nvidia-smi", shell=True).decode()
        return "GPU is available", out
    except subprocess.CalledProcessError:
        return "GPU is not available", ""

print(*check_gpu())

# 讀取 PDF QA
def load_reference_answers_from_pdf(pdf_path):
    """
    從PDF中讀取Q:C:A:格式的題目
    返回格式: [(question, answer, chinese_translation), ...]
    如果找不到Q:C:A:格式，會回退到原始格式
    """
    doc = fitz.open(pdf_path)
    lines = []
    
    # 收集所有行
    for page in doc:
        for line in page.get_text().split("\n"):
            line = line.strip()
            if line and line.startswith(("Q:", "C:", "A:")):
                lines.append(line)
    
    doc.close()
    
    # 解析Q:C:A:格式
    qa_pairs = []
    i = 0
    while i < len(lines):
        if i + 2 < len(lines) and (
            lines[i].startswith("Q:") and 
            lines[i+1].startswith("C:") and 
            lines[i+2].startswith("A:")):
            
            question = lines[i][2:].strip()
            chinese = lines[i+1][2:].strip()
            answer = lines[i+2][2:].strip()
            
            qa_pairs.append((question, answer, chinese))
            i += 3
        else:
            i += 1
    
    if not qa_pairs:
        print("警告：沒有找到Q:C:A:格式，使用原始格式解析")
        # 重新打開文件，使用原始邏輯
        doc = fitz.open(pdf_path)
        qas = []
        for page in doc:
            for line in page.get_text().split("\n"):
                if line.startswith(("Q:", "A:")):
                    qas.append(line.strip())
        doc.close()
        
        qs, ans = [], []
        for i in range(0, len(qas)-1, 2):
            if qas[i].startswith("Q:") and qas[i+1].startswith("A:"):
                qs.append(qas[i][2:].strip())
                ans.append(qas[i+1][2:].strip())
        
        # 返回三元組格式，中文翻譯為空字符串
        qa_pairs = [(q, a, "") for q, a in zip(qs, ans)]
    
    return qa_pairs

# 解析單題得分
score_re = re.compile(r"整體表現評分[:：]\s?(\d(?:\.\d)?)\s?分")

def parse_result(txt):
    m = score_re.search(txt)
    return float(m.group(1)) if m else 0

# 單題評分 (呼叫 ollama)

def evaluate_single_answer(ans, q, ref):
    prompt = f"""
你是一位英文老師，請依下列「評分標準」對學生的口說回答進行打分並給回饋。  
學生皆是初級程度，評分無需過多嚴格。
請用繁體中文回答，格式一定要包含：  
1. 整體表現評分（0-5 分，必須是整數）  
2. 錯誤說明與改善建議  
3. 參考答案（可簡短列出重點）

【評分標準】
5分:發音清晰、正確；語調自然。內容切題，表達流暢；語法與字彙偶有小錯誤但不影響溝通。  
4分:發音、語調大致正確；少數錯誤。內容切題；語法、字彙偶有錯誤但不影響溝通。  
3分:發音/語調時有錯誤，略影響理解；基本句型可用，但語法、字彙不足以完整表達。  
2分:發音/語調錯誤偏多；朗讀時常跳過難字；語法、字彙錯誤造成溝通困難。  
1分:發音/語調嚴重錯誤；句構錯亂，單字量嚴重不足，幾乎無法溝通。  
0分:未答或內容與題目無關。

【題目】{q}
【參考答案】{ref}
【學生回答】{ans}
"""
    res = ollama.chat(
        model="gemma3:12b",
        messages=[{"role": "user", "content": prompt}],
    )
    return res["message"]["content"]

# 綜合評語

def overall_comment(records):
    total = sum(r["score"] for r in records)
    avg   = total / len(records)
    detail = "\n\n".join([f"題目{i+1}:\n{r['result']}" for i,r in enumerate(records)])
    prompt = f"""
你是英文老師，學生平均 {avg:.2f}/5 分，
請用 80 字內給中文鼓勵式綜合評語。(請勿將提示語或字詞顯示出來)
評語的開頭請用:同學好，
細節：\n{detail}"""
    res = ollama.chat(model="gemma3:12b", messages=[{"role":"user","content":prompt}])
    return avg, res["message"]["content"]


# JSON 檔案路徑
SCORES_FILE = 'test_scores.json'

def load_scores():
    """
    從 JSON 檔案載入分數紀錄
    """
    try:
        if os.path.exists(SCORES_FILE):
            with open(SCORES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"載入分數檔案時發生錯誤: {e}")
        return []

def save_scores(scores):
    """
    將分數紀錄儲存到 JSON 檔案
    """
    try:
        with open(SCORES_FILE, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"儲存分數檔案時發生錯誤: {e}")

def add_test_score(score):
    """
    新增測驗分數到紀錄中
    """
    scores = load_scores()
    
    # 新增分數記錄
    score_record = {
        'score': round(score, 2),
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    scores.append(score_record)
    
    # 只保留最近100筆記錄以避免檔案過大
    if len(scores) > 100:
        scores = scores[-100:]
    
    save_scores(scores)

def get_recent_records(limit=5):
    """
    取得最近的測驗記錄
    """
    try:
        scores = load_scores()
        
        if not scores:
            return []
        
        # 取得最近的記錄
        recent_records = scores[-limit:] if len(scores) >= limit else scores
        
        # 反轉順序，讓最新的記錄顯示在最上面
        return list(reversed(recent_records))
        
    except Exception as e:
        print(f"取得最近記錄時發生錯誤: {e}")
        return []

def get_recent_average_score():
    """
    取得最近五筆紀錄的平均分數
    """
    try:
        scores = load_scores()
        
        if not scores:
            return None
        
        # 取得最近的5筆紀錄
        recent_scores = scores[-5:] if len(scores) >= 5 else scores
        
        if recent_scores:
            total_score = sum(record['score'] for record in recent_scores)
            average = round(total_score / len(recent_scores), 1)
            return average
        
        return None
        
    except Exception as e:
        print(f"取得平均分數時發生錯誤: {e}")
        return None

# -------------------------------------------------------------
#  Flask Routes
# -------------------------------------------------------------
QUIZ_QUESTION_COUNT = 5

@app.route("/")
def home():
    # session.clear()
    recent_records = get_recent_records(5)
    return render_template("index.html", recent_records=recent_records)

@app.route("/feature")
def feature():
    return render_template("feature.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/example")
def example():
    return render_template("example.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    # 如果出現問題，輸出一條消息
    msg = ''
    # 檢查 POST 請求中是否存在 "username", "password" 和 "email"（用戶提交了表單）
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # 創建變量以便於訪問
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # 使用 MySQL 檢查帳戶是否存在
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # 如果帳戶存在，顯示錯誤並進行驗證檢查
        if account:
            msg = '帳戶已存在！'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = '無效的電子郵件地址！'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = '用戶名只能包含字母和數字！'
        elif not username or not password or not email:
            msg = '請填寫表單！'
        else:
            # 帳戶不存在且表單數據有效，現在將新帳戶插入到 accounts 表中
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = '您已成功註冊！'
    elif request.method == 'POST':
        # 表單為空...（沒有 POST 數據）
        msg = '請填寫表單！'
    # 顯示註冊表單並顯示消息（如果有）
    return render_template('register.html', msg=msg)

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'loggedin' in session and session['loggedin']:
        return redirect(url_for('voice_select'))
    # 如果出現問題，輸出一條消息...
    msg = ''
    
    # 檢查 POST 請求中是否存在 "email" 和 "password"（用戶提交了表單）
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        
        # 使用 MySQL 檢查帳戶是否存在
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE email = %s AND password = %s', (email, password,))
        # 獲取一條記錄並返回結果
        account = cursor.fetchone()
        
        # 如果帳戶存在於資料庫中的 accounts 表
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['email'] = account['email']  # 儲存email而不是username
            return redirect(url_for('home'))
        else:
            # 帳戶不存在或email/密碼不正確
            msg = '電子郵件或密碼不正確！'
    
    # 顯示登錄表單並顯示消息（如果有）
    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   return redirect(url_for('login'))

@app.route('/profile')
def profile():
    # 檢查用戶是否已登錄
    if 'loggedin' in session:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        return render_template('profile.html', account=account)
    # 用戶未登錄，重定向到登錄頁面
    return redirect(url_for('login'))


@app.route("/voice_select", methods=["GET", "POST"])
def voice_select():
    # 檢查用戶是否已登錄
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    if request.method == "GET":
        return render_template("voice_select.html")

    voice = request.form.get("voice_model", "mann")
    session["voice_model"] = voice
    return redirect(url_for("english"))

@app.route("/english", methods=["GET", "POST"])
def english():
    pdf_path = "GEPT_Complete.pdf"
    if not os.path.exists(pdf_path):
        return "找不到教材 PDF"

    # 使用新的解析函數，返回包含中文翻譯的格式
    qa_pairs = load_reference_answers_from_pdf(pdf_path)
    session.setdefault("records", [])
    session.setdefault("used_idx", [])
    session.setdefault("q_no", 1)

    if "current_q" not in session:
        idx = random.randrange(len(qa_pairs))
        session["current_q"] = qa_pairs[idx]  # 現在包含(question, answer, chinese)
        session["used_idx"].append(idx)

    if request.method == "POST":
        user_ans = request.form.get("user_answer", "").strip()
        
        # 安全地解包，兼容舊格式
        current_q = session["current_q"]
        if len(current_q) == 3:
            q, ref, chinese = current_q
        else:
            q, ref = current_q
            chinese = ""
            
        if user_ans:
            result = evaluate_single_answer(user_ans, q, ref)
            score  = parse_result(result)
            session["records"].append({"question":q,"answer":user_ans,"result":result,"score":score})
        
        if request.form.get("action") == "finish" or session["q_no"] >= QUIZ_QUESTION_COUNT:
            return redirect(url_for("eng_result"))
        
        session["q_no"] += 1
        remain = [i for i in range(len(qa_pairs)) if i not in session["used_idx"]]
        if remain:  # 確保還有剩餘的題目
            idx = random.choice(remain)
            session["current_q"] = qa_pairs[idx]
            session["used_idx"].append(idx)

    # 安全地解包，兼容舊格式
    current_q = session["current_q"]
    if len(current_q) == 3:
        q, ref, chinese_translation = current_q
    else:
        q, ref = current_q
        chinese_translation = ""
        
    last = session["q_no"] >= QUIZ_QUESTION_COUNT
    
    return render_template("english.html", 
                         question=q, 
                         chinese_translation=chinese_translation,
                         show_next=not last, 
                         show_end=last)

@app.route("/eng_result")
def eng_result():
    if "records" not in session:
        return redirect(url_for("home"))

    while len(session["records"]) < QUIZ_QUESTION_COUNT:
        session["records"].append({"question":"未作答","answer":"未作答","result":"無資料","score":0})

    avg, comment = overall_comment(session["records"])
    add_test_score(avg)

    # 只朗讀平均分與綜合評語
    try:
        tts(f"你的平均分數是 {avg:.2f} 分。{comment}", session.get("voice_model", "mann"),out_path="static/output.wav")
    except Exception as e:
        print("TTS 失敗", e)
    return render_template("eng_result.html", records=session["records"], avg_score=avg, overall_evaluation=comment)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)#"120.105.129.156",