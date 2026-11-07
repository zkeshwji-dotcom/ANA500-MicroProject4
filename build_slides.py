"""Fills the ANA500 micro-project template with the energy demand forecasting results and figures."""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

TEMPLATE = "/Users/zeenakeshwji/Downloads/ANA500_Micro-Project_template (2).pptx"
OUT = "ANA500_MicroProject4_Energy_Demand.pptx"
CORAL, TEAL, INK, MUTED = RGBColor(0xE4, 0x57, 0x2E), RGBColor(0x17, 0x87, 0x7B), RGBColor(0x26, 0x2B, 0x33), RGBColor(0x5F, 0x66, 0x70)
TINT = RGBColor(0xE8, 0xF3, 0xF1)

prs = Presentation(TEMPLATE)
content_layout = prs.slides[1].slide_layout


def move_slide(slide, new_index):
    lst = prs.slides._sldIdLst
    el = [s for s in lst if s.get("id") == str(slide.slide_id)][0]
    lst.remove(el)
    lst.insert(new_index, el)


def add_after(index, title):
    s = prs.slides.add_slide(content_layout)
    s.shapes.title.text = title
    move_slide(s, index + 1)
    return s


def body_of(slide):
    return [ph for ph in slide.placeholders if ph.placeholder_format.idx == 1][0]


def set_title(slide, text):
    tf = slide.shapes.title.text_frame
    p = tf.paragraphs[0]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    p.runs[0].text = text
    for extra in tf.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)


def fill_bullets(slide, items, left, top, width, height, size=16):
    """items: list of (text, level) or (label, text, level) where label is bold."""
    ph = body_of(slide)
    ph.left, ph.top, ph.width, ph.height = Inches(left), Inches(top), Inches(width), Inches(height)
    tf = ph.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if len(item) == 3:
            label, text, level = item
        else:
            label, (text, level) = None, item
        p.level = level
        p.space_after = Pt(6)
        if label:
            r = p.add_run(); r.text = label + " "; r.font.bold = True; r.font.size = Pt(size - 2 * level); r.font.color.rgb = INK
        r = p.add_run(); r.text = text; r.font.size = Pt(size - 2 * level); r.font.color.rgb = INK
    return ph


def picture(slide, path, left, top, width=None, height=None, alt=""):
    pic = slide.shapes.add_picture(path, Inches(left), Inches(top),
                                   Inches(width) if width else None, Inches(height) if height else None)
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def caption(slide, text, left, top, width):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.35))
    tf = tb.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = 0
    r = tf.paragraphs[0].add_run(); r.text = text; r.font.size = Pt(11); r.font.italic = True; r.font.color.rgb = MUTED
    return tb


def table(slide, rows, left, top, width, col_widths, size=12, highlight_col=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top), Inches(width), Inches(0.38 * len(rows)))
    t = shape.table
    for j, w in enumerate(col_widths):
        t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(val)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(size)
            if i == 0:
                p.font.bold = True; p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid(); cell.fill.fore_color.rgb = TEAL
            else:
                p.font.color.rgb = INK
                cell.fill.solid(); cell.fill.fore_color.rgb = TINT if i % 2 else RGBColor(0xFF, 0xFF, 0xFF)
                if highlight_col is not None and j == highlight_col and val in ("Yes", "Supported"):
                    p.font.bold = True; p.font.color.rgb = TEAL
    return shape


def stat(slide, number, label, left, top, width, color):
    box = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(1.45))
    box.fill.solid(); box.fill.fore_color.rgb = TINT; box.line.fill.background(); box.shadow.inherit = False
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.12)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = number; r.font.size = Pt(36); r.font.bold = True; r.font.color.rgb = color
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r = p2.add_run(); r.text = label; r.font.size = Pt(12); r.font.color.rgb = INK
    return box




# ---- Structure first: continuation slides after Prepare, Analyze and Report ----
add_after(4, "Prepare (cont.): Features and Windows")
add_after(6, "Analyze Data (cont.): Results")
add_after(8, "Report (cont.): Where Errors Fall")
s = prs.slides

# ---- 1. Title ----
t = s[0]
tf = t.shapes.title.text_frame
runs = [r for p in tf.paragraphs for r in p.runs]
runs[0].text = "Forecasting Electricity Demand"
runs[0].font.size = Pt(44); runs[0].font.bold = True
runs[1].text = "Micro-Project 4"
runs[1].font.size = Pt(28)
runs[2].text = "Repository: github.com/zkeshwji-dotcom/ANA500-MicroProject4"
runs[2].hyperlink.address = "https://github.com/zkeshwji-dotcom/ANA500-MicroProject4"
runs[2].font.size = Pt(20); runs[2].font.color.rgb = MUTED
for br in tf.paragraphs[0]._p.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}br"):
    rpr = br.find("{http://schemas.openxmlformats.org/drawingml/2006/main}rPr")
    if rpr is not None:
        rpr.set("sz", "2800")
sub = t.placeholders[1].text_frame
sub.paragraphs[0].runs[0].text = "Zeena Keshwji"
sub.paragraphs[1].runs[0].text = "November 6, 2026"
sub.paragraphs[2]._p.getparent().remove(sub.paragraphs[2]._p)

# ---- 2. Problem statement ----
sl = s[1]
fill_bullets(sl, [
    ("Electricity is hard to store at grid scale, so supply must match demand hour by hour.", 0),
    ("PJM East load swings between 14,500 MW and 62,000 MW.", 0),
    ("Forecasting too low risks shortages and costly last-minute generation; too high wastes committed capacity.", 0),
    ("Question:", "How accurately can deep learning predict each of the next 24 hourly loads from past load and the calendar?", 0),
], 0.6, 1.75, 5.4, 4.9)
picture(sl, "figures/01_load_overview.png", 6.25, 1.9, width=6.6, alt="Daily peak load from 2002 to 2018 and one week of hourly load in July 2017")
caption(sl, "Source: Kaggle, Hourly Energy Consumption (robikscube), PJM East region", 6.25, 5.2, 6.6)

# ---- 3. Hypothesis ----
sl = s[2]
fill_bullets(sl, [
    ("Demand is a repeating pattern with memory, so a model that reads load as a sequence should forecast it best.", 0),
    ("H1:", "Load has strong daily, weekly and annual cycles.", 0),
    ("H2:", "A recurrent network beats naive and linear baselines.", 0),
    ("H3:", "Gated RNNs (LSTM, GRU) beat a plain RNN and a dense network.", 0),
    ("H4:", "One week of history beats one day.", 0),
    ("H5:", "The largest errors fall on summer peaks and holidays, since there is no weather data.", 0),
], 0.6, 1.75, 5.9, 4.9, size=15)
table(sl, [
    ["Hypothesis", "Planned test"],
    ["H1", "Autocorrelation, calendar profiles"],
    ["H2", "Paired block bootstrap on daily MAE"],
    ["H3", "Paired block bootstrap on daily MAE"],
    ["H4", "Each model at 24 h vs. 168 h window"],
    ["H5", "Mann-Whitney test by day type"],
], 6.9, 2.0, 5.9, [1.5, 4.4], size=13)
caption(sl, "All tests at α = 0.05 on a chronological hold-out set", 6.9, 4.45, 5.9)

# ---- 4. Acquire ----
sl = s[3]
fill_bullets(sl, [
    ("Identify:", "Kaggle “Hourly Energy Consumption” (robikscube): load published by PJM Interconnection, a U.S. grid operator.", 0),
    ("Retrieve:", "Downloaded the ZIP from Kaggle's public API; the notebook re-downloads it if the local copy is missing.", 0),
    ("Query:", "Read PJME_hourly.csv, the longest regional series, into pandas: 145,366 rows × 2 columns.", 0),
    ("Target:", "the next 24 hourly loads in MW, all continuous, so this is multi-output regression.", 0),
], 0.6, 1.75, 5.9, 4.9)
table(sl, [
    ["Item", "Value"],
    ["Column 1", "Datetime (hourly timestamp)"],
    ["Column 2", "PJME_MW (load, megawatts)"],
    ["Period", "Jan 1, 2002 to Aug 3, 2018"],
    ["Mean load", "32,079 MW"],
    ["Lowest", "14,544 MW (Hurricane Sandy, 2012)"],
    ["Highest", "62,009 MW (Aug 2, 2006)"],
], 6.9, 1.9, 5.9, [1.6, 4.3], size=12)
caption(sl, "kaggle.com/datasets/robikscube/hourly-energy-consumption", 6.9, 4.8, 5.9)

# ---- 5. Prepare ----
sl = s[4]
fill_bullets(sl, [
    ("Rows were out of order: sorted by time.", 0),
    ("8 rows shared a timestamp (fall clock change): averaged.", 0),
    ("30 single hours were missing, mostly the spring clock change: interpolated.", 0),
    ("Result: a complete grid of 145,392 hours.", 0),
    ("H1 supported:", "peak hour is 43% above the overnight low, July is 36% above April, and weekdays run 11% above weekends.", 0),
], 0.6, 1.75, 5.0, 4.9, size=15)
picture(sl, "figures/02_calendar_patterns.png", 5.85, 2.2, width=7.0, alt="Heatmap of average load by month and hour, and average daily shape on weekdays and weekends")
caption(sl, "Demand peaks on summer afternoons and on winter evenings", 5.85, 4.85, 7.0)

# ---- 6. Prepare (cont.) ----
sl = s[5]
fill_bullets(sl, [
    ("Inputs:", "past load plus calendar features (sine and cosine of hour, weekday and day of year; weekend or holiday flag).", 0),
    ("Known in advance:", "the calendar of the next 24 hours.", 0),
    ("Scaling:", "load standardized with training statistics only.", 0),
    ("Chronological split, no shuffling:", "", 0),
], 0.6, 1.75, 6.0, 2.9, size=15)
table(sl, [
    ["Set", "Dates", "Forecasts"],
    ["Train", "2002 to 2014", "18,960 (every 6 h)"],
    ["Validation", "2015 to 2016", "731 (daily)"],
    ["Test", "Jan 2017 to Aug 2018", "579 (daily)"],
], 0.6, 4.7, 6.0, [1.5, 2.5, 2.0], size=13)
picture(sl, "figures/03_autocorrelation.png", 6.9, 2.0, width=5.95, alt="Autocorrelation of load out to two weeks with peaks at 24 and 168 hours")
caption(sl, "Load correlates 0.89 with the same hour yesterday and 0.78 with the same hour last week", 6.9, 4.45, 5.95)

# ---- 7. Analyze ----
sl = s[6]
fill_bullets(sl, [
    ("Baselines:", "same hour yesterday, same hour last week (seasonal naive), ridge regression, gradient boosting.", 0),
    ("Deep learning (Keras):", "dense network (MLP), Simple RNN, LSTM, GRU, Seq2Seq LSTM.", 0),
    ("Fair comparison:", "every trained model tried 24 h and 168 h windows and kept the better one on validation.", 0),
    ("Training:", "Adam, MSE loss, early stopping, 3 seeds per network, forecasts averaged across seeds.", 0),
    ("Iteration:", "the first run was redone after finding unequal windows and an epoch cap.", 0),
], 0.6, 1.75, 12.1, 2.75, size=15)
picture(sl, "figures/04_learning_curves.png", 0.9, 4.5, width=11.5, alt="Training and validation loss curves for the five neural networks")

# ---- 8. Analyze (cont.) ----
sl = s[7]
table(sl, [
    ["Model", "Window", "Val MAE", "Test MAE", "Test MAPE", "Test R²"],
    ["Simple RNN (final)", "168 h", "1,151", "1,337", "4.1%", "0.902"],
    ["MLP", "24 h", "1,160", "1,331", "4.1%", "0.900"],
    ["GRU", "168 h", "1,165", "1,359", "4.2%", "0.900"],
    ["Seq2Seq LSTM", "24 h", "1,190", "1,374", "4.3%", "0.898"],
    ["LSTM", "24 h", "1,203", "1,412", "4.4%", "0.893"],
    ["Ridge", "168 h", "1,245", "1,378", "4.2%", "0.892"],
    ["Gradient boosting", "24 h", "1,248", "1,416", "4.4%", "0.888"],
    ["Same hour yesterday", "n/a", "2,137", "2,300", "7.3%", "0.738"],
    ["Seasonal naive", "n/a", "3,312", "3,511", "11.0%", "0.387"],
], 0.6, 1.85, 7.3, [2.55, 0.9, 0.95, 1.0, 1.05, 0.85], size=12)
caption(sl, "MAE in MW. Final model chosen on validation MAE; the MLP's small test lead is not significant.", 0.6, 5.85, 7.3)
table(sl, [
    ["Test-set comparison", "MAE gain", "p"],
    ["Simple RNN vs. naive", "2,174", "< 0.001"],
    ["Simple RNN vs. boosting", "79", "< 0.001"],
    ["Simple RNN vs. ridge", "40", "0.12"],
    ["Simple RNN vs. MLP", "−6", "0.53"],
    ["Simple RNN vs. GRU", "22", "0.014"],
    ["Simple RNN vs. LSTM", "75", "< 0.001"],
], 8.2, 1.85, 4.65, [2.65, 1.0, 1.0], size=12)
caption(sl, "Paired block bootstrap on daily MAE, 14-day blocks", 8.2, 4.65, 4.65)

# ---- 9. Report ----
sl = s[8]
body_of(sl)._element.getparent().remove(body_of(sl)._element)   # this slide uses stat boxes and a table instead
stat(sl, "4.1%", "average error of the Simple RNN (1,337 MW)", 0.6, 1.8, 2.3, TEAL)
stat(sl, "−62%", "error vs. the seasonal-naive forecast", 3.05, 1.8, 2.3, TEAL)
table(sl, [
    ["Hypothesis", "Verdict"],
    ["H1 Strong cycles", "Supported"],
    ["H2 RNN beats baselines", "Partly: tie with ridge"],
    ["H3 Gated RNNs win", "Not supported"],
    ["H4 One week beats one day", "Partly: ridge, GRU only"],
    ["H5 Errors at peaks, holidays", "Partly: peaks only"],
], 0.6, 3.5, 4.75, [2.6, 2.15], size=12, highlight_col=1)
picture(sl, "figures/06_forecast_examples.png", 5.6, 1.8, width=7.25, alt="Forecast against actual load for a summer week and a winter week")
caption(sl, "The RNN tracks the daily shape closely but runs low during the January 2018 cold snap", 5.6, 5.55, 7.25)

# ---- 10. Report (cont.) ----
sl = s[9]
fill_bullets(sl, [
    ("Summer error is highest:", "1,646 MW vs. 1,099 to 1,320 MW in other seasons (p < 0.001).", 0),
    ("Top 5% peak days:", "1,777 MW vs. 1,311 MW on normal days (p < 0.001).", 0),
    ("Holidays:", "no detectable difference (p = 0.18, only 15 days).", 0),
    ("Worst five days:", "all over-forecasts after a sharp drop in peak load, which fits sudden weather changes, though the data cannot confirm this.", 0),
    ("Error grows with lead time:", "274 MW at 1 hour, 1,737 MW at 24 hours.", 0),
], 0.6, 1.75, 5.3, 4.9, size=15)
picture(sl, "figures/07_error_structure.png", 6.1, 2.2, width=6.75, alt="Mean absolute error by month and by hour of day")
caption(sl, "Error is highest in June through August and in the late afternoon and evening", 6.1, 4.6, 6.75)

# ---- 11. Act ----
sl = s[10]
fill_bullets(sl, [
    ("Use the Simple RNN:", "lowest validation error, smallest network, trains in about 2 minutes. Keep ridge as a transparent backup.", 0),
    ("Size reserves from measured error:", "about 4,500 MW above forecast covered 94.5% of test peaks, vs. about 9,400 MW for naive. Ridge needs a similar 4,586 MW.", 0),
    ("Recalibrate bands:", "the 90% band covered only 86.6% of hours (81% in summer).", 0),
    ("Add weather next:", "it is the most likely source of the remaining error.", 0),
    ("Limits:", "one region, data ends in 2018, and forecasts are issued at midnight, later than a real day-ahead market.", 0),
], 0.6, 1.75, 6.2, 5.0, size=14)
picture(sl, "figures/08_peak_day_band.png", 7.0, 2.0, width=5.85, alt="Forecast, 90 percent band and actual load on July 2, 2018")
caption(sl, "Highest-demand test day: the forecast peak came in 4.1% low", 7.0, 4.95, 5.85)

prs.save(OUT)
print("Saved", OUT, "with", len(prs.slides), "slides")
