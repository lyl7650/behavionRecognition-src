import json, glob, os
base = "dataset/raw_videos/rabbit/hkbr-record/records"
for f in glob.glob(os.path.join(base, "*", ".autopresort", "*.json")):
    if "163024_b56f8160" in f:
        doc = json.load(open(f, encoding="utf-8"))
        out = {"rel_path":doc["rel_path"],"duration":doc["duration"],"approved":doc["approved"],
               "segments":doc["segments"],"segment_levels":doc["segment_levels"],"meta":doc.get("meta",{})}
        open("auto_presort/_realdoc.json","w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False))
        # compute expected percents for seg[0]
        D = doc["duration"]
        for s in doc["segments"]:
            print(f"  {s['label']}: left={s['start']/D*100:.1f}% width={(s['end']-s['start'])/D*100:.1f}%")
        break
