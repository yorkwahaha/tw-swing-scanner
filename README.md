# 短線雷達 · 臺股 1–2 週波段觀察清單

一個會「每天自動跑」的網頁:收盤後用固定規則從全上市股票裡篩出一份
技術面觀察清單,存成靜態網頁給你每天看。全部用免費公開資料,免主機費用
(掛在 GitHub 上)。

**這不是選股神器,也不是投資建議。** 篩選邏輯就是均線、RSI、量能、相對
大盤強弱這幾個很常見的指標組合而成,沒有做過嚴謹回測,過去價量表現也
不保證未來報酬。把它當成「幫你先縮小範圍的清單」,實際進出場還是要靠
你自己判斷。

---

## 架構

```
scanner.py          <- 每天跑一次的資料流程(見下方「怎麼運作」)
docs/index.html      <- 前端網頁(讀取 docs/data.json 顯示排行榜)
docs/data.json        <- scanner.py 產生的結果,前端讀這個檔案
.github/workflows/daily.yml  <- 排程:收盤後自動跑 scanner.py 並發布
requirements.txt      <- Python 套件需求
```

### 怎麼運作(四階段)

1. **全市場快照** — 打一次 TWSE 官方 OpenAPI(`STOCK_DAY_ALL`),拿到當天
   所有上市股票的收盤價、漲跌、成交量,不用對每檔股票各打一次 API。
2. **流動性初篩** — 用今天的成交量、股價區間先把幾千檔縮小到候選池
   (預設約 300 檔),避免下一步打爆 API。
3. **抓歷史資料算指標** — 對候選池用 yfinance 抓近 90 個交易日的日K,
   計算 5/20 日均線、RSI(14)、量能比、相對大盤 5 日強弱。
4. **合成分數並輸出** — 依權重合成 0–100 分,排序後取前 N 檔寫進
   `docs/data.json`,前端網頁讀這個檔案顯示。

---

## 快速開始:部署到 GitHub(全自動每天更新)

1. **建立 repo**:把這個資料夾整個推到你自己的 GitHub repo(public 或
   private 皆可,private 的話 Pages 需要付費方案才能開,建議用 public)。

2. **開啟 GitHub Pages**:repo 的 `Settings → Pages`,Source 選
   `Deploy from a branch`,Branch 選 `main`、資料夾選 `/docs`,存檔。
   之後網址會長得像 `https://你的帳號.github.io/repo名稱/`。

3. **開放 Actions 寫入權限**:repo 的 `Settings → Actions → General →
   Workflow permissions`,選 `Read and write permissions`,存檔。
   (這樣排程跑完才能把新的 `data.json` commit 回 repo。)

4. **手動測試一次**:repo 的 `Actions` 分頁 → 選 `每日掃描臺股短線觀察清單`
   → `Run workflow`,跑完檢查 `docs/data.json` 有沒有更新、Pages 網頁
   有沒有正常顯示。

5. 之後平日(週一~五)台北時間 14:30 會自動跑一次。想改時間就改
   `.github/workflows/daily.yml` 裡的 `cron`(注意 GitHub Actions 的
   cron 是 UTC 時間)。

---

## 本機測試

```bash
pip install -r requirements.txt

# 離線 demo,不會打任何 API,純粹看畫面長怎樣
python scanner.py --demo --top 20

# 正式模式,會真的打 TWSE + yfinance(交易日跑才有意義)
python scanner.py --top 20
```

跑完後用瀏覽器打開 `docs/index.html`(或起一個簡單的本機伺服器
`python -m http.server -d docs`)就能看到畫面。

---

## 篩選邏輯與評分

| 指標 | 權重 | 說明 |
|---|---|---|
| 趨勢 | 30% | 站上 20 日均線,且 5 日均線在 20 日均線之上 |
| 動能 | 25% | RSI(14) 落在 50–70 健康偏強區間;超過 80 視為短線過熱、扣分 |
| 量能 | 20% | 今日成交量 ÷ 20 日均量,放大越多分數越高 |
| 相對強弱 | 25% | 近 5 日報酬 減去 大盤(加權指數)近 5 日報酬 |

四項加權合成 0–100 分,分數越高代表越多指標同時偏多。清單上的標籤
(如「5/20MA黃金交叉」「成交量放大」「強於大盤」「短線過熱」)是每檔
股票觸發了哪些條件的白話說明,不是額外加分項以外的東西。

---

## 客製化

- **調整篩選門檻 / 權重**:都在 `scanner.py` 最上面的參數區
  (`MIN_TRADE_VOLUME_SHARES`、`WEIGHTS` 等),改完直接重跑即可,不用動
  邏輯本身。
- **想涵蓋上櫃(TPEx)股票**:目前 `STOCK_DAY_ALL` 只涵蓋上市,上櫃股票
  要另外接 TPEx OpenAPI 的對應端點,是同樣的邏輯但欄位命名不同。
- **想換資料來源**:如果 yfinance 抓資料不穩定,可以換成
  [FinMind](https://finmind.github.io/) 的 `TaiwanStockPrice` 資料集
  (免費額度 300–600 次/小時,註冊後更高)。

---

## 限制與注意事項

- 這是「收盤後」的每日批次更新,不是即時盤中資料,不適合當沖使用。
- 篩選邏輯沒有做過回測,權重是憑經驗設的合理起點,不是最佳化結果。
- yfinance 抓取的資料偶爾會有缺漏或延遲,`scanner.py` 對抓不到足夠
  歷史資料的股票會直接跳過,不會硬湊資料。
- 這份清單只做技術面規則篩選,不參考基本面、籌碼面(如三大法人買賣超)
  或消息面,使用前請自行評估這些面向。
- 我沒有帳號權限或即時網路環境幫你把這個 repo 實際部署上線並跑過真實
  資料,所以請照上面步驟自己跑一次 `workflow_dispatch` 確認欄位對得起來
  (TWSE API 的欄位命名偶爾會微調,`scanner.py` 在對不到預期欄位時會在
  Actions 的 log 裡印警告,方便你排查)。

**再次提醒:這不是投資建議。** 短線交易風險本來就高,任何買賣決定都是
你自己的判斷與責任。
