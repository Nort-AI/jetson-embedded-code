import os, re, json

locales_file = r'c:/Users/enzog/PycharmProjects/PYTHON/NortV0.1.3/Jetson-Embedded-code-3/admin/locales/en.json'
with open(locales_file, encoding='utf-8') as f:
    en_data = json.load(f)

html_dir = r'c:/Users/enzog/PycharmProjects/PYTHON/NortV0.1.3/Jetson-Embedded-code-3/admin/templates'
keys = set()
for f in os.listdir(html_dir):
    if f.endswith('.html'):
        content = open(os.path.join(html_dir, f), encoding='utf-8').read()
        keys.update(re.findall(r"t\('([a-zA-Z0-9_\.]+)'\)", content))

defined_keys = set()
for k, v in en_data.items():
    if isinstance(v, str):
        defined_keys.add(k)
    elif isinstance(v, dict):
        for k2 in v.keys():
            defined_keys.add(f'{k}.{k2}')

missing = sorted(list(keys - defined_keys))
print("MISSING KEYS:")
for m in missing:
    print(m)
