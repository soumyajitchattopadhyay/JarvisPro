"""Download 1000 popular brand icons into icons/ and write manifest.json."""
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON_DIR = os.path.join(ROOT, "icons")
MANIFEST_PATH = os.path.join(ICON_DIR, "manifest.json")
TARGET_COUNT = 1000
ICON_COLOR = "#00fbff"
SVG_BASE = "https://cdn.jsdelivr.net/npm/simple-icons@11.15.0/icons"
INDEX_URL = "https://data.jsdelivr.com/v1/package/npm/simple-icons@11.15.0/flat"

# Popular icons first — tech, dev tools, languages, platforms, gaming, etc.
PRIORITY = [
    "openai", "google", "github", "python", "tesla", "nvidia", "react", "rust", "apple", "microsoft",
    "amazon", "meta", "netflix", "spotify", "discord", "slack", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "gitlab", "bitbucket", "jira", "confluence", "notion", "figma", "adobe",
    "photoshop", "illustrator", "premierepro", "aftereffects", "blender", "unity", "unrealengine",
    "steam", "epicgames", "playstation", "xbox", "nintendo", "roblox", "minecraft", "valorant",
    "leagueoflegends", "fortnite", "pubg", "dota2", "counterstrike", "twitch", "youtube", "tiktok",
    "instagram", "facebook", "twitter", "x", "linkedin", "reddit", "pinterest", "snapchat", "whatsapp",
    "telegram", "signal", "zoom", "webex", "teams", "dropbox", "box", "onedrive", "icloud", "mega",
    "aws", "googlecloud", "azure", "digitalocean", "cloudflare", "vercel", "netlify", "heroku",
    "railway", "render", "supabase", "firebase", "mongodb", "postgresql", "mysql", "mariadb",
    "redis", "elasticsearch", "kafka", "rabbitmq", "nginx", "apache", "caddy", "traefik", "haproxy",
    "linux", "ubuntu", "debian", "fedora", "archlinux", "kalilinux", "android", "ios", "flutter",
    "dart", "kotlin", "swift", "objectivec", "java", "spring", "gradle", "maven", "nodejs",
    "deno", "bun", "npm", "yarn", "pnpm", "webpack", "rollup", "vite", "parcel", "esbuild",
    "babel", "eslint", "prettier", "typescript", "javascript", "html5", "css3", "sass", "less",
    "tailwindcss", "bootstrap", "materialdesign", "mui", "chakraui", "antdesign", "storybook",
    "nextdotjs", "nuxt", "remix", "astro", "gatsby", "svelte", "vue", "angular", "emberdotjs",
    "solid", "qwik", "htmx", "alpinejs", "jquery", "express", "fastapi", "django", "flask",
    "rails", "laravel", "symfony", "springboot", "dotnet", "csharp", "cplusplus", "c", "go",
    "ruby", "php", "perl", "haskell", "scala", "clojure", "elixir", "erlang", "lua", "r",
    "julia", "matlab", "octave", "fortran", "assembly", "zig", "nim", "crystal", "v", "odin",
    "tensorflow", "pytorch", "keras", "scikitlearn", "pandas", "numpy", "jupyter", "anaconda",
    "opencv", "huggingface", "ollama", "langchain", "openjdk", "intel", "amd", "qualcomm",
    "arm", "raspberrypi", "arduino", "espressif", "nvidia", "cisco", "juniper", "paloaltonetworks",
    "fortinet", "crowdstrike", "sentinelone", "splunk", "datadog", "grafana", "prometheus",
    "influxdb", "opentelemetry", "sentry", "newrelic", "dynatrace", "pagerduty", "hashicorp",
    "vault", "consul", "nomad", "pulumi", "circleci", "travisci", "githubactions", "gitlabci",
    "sonarqube", "codecov", "snyk", "dependabot", "postman", "insomnia", "swagger", "openapiinitiative",
    "graphql", "prisma", "sequelize", "typeorm", "drizzle", "sqlite", "duckdb", "snowflake",
    "databricks", "apachespark", "apachehadoop", "apacheairflow", "dbt", "tableau", "powerbi",
    "looker", "metabase", "obsidian", "logseq", "roamresearch", "evernote", "todoist", "trello",
    "asana", "linear", "clickup", "monday", "basecamp", "airtable", "coda", "miro", "canva",
    "sketch", "invision", "framer", "webflow", "squarespace", "wix", "shopify", "woocommerce",
    "stripe", "paypal", "square", "venmo", "cashapp", "coinbase", "binance", "ethereum", "bitcoin",
    "solana", "polygon", "chainlink", "opensea", "metamask", "ledger", "trezor", "visa", "mastercard",
    "americanexpress", "discover", "chase", "bankofamerica", "wellsfargo", "goldmansachs", "jpmorgan",
    "morganstanley", "blackrock", "fidelity", "robinhood", "revolut", "wise", "nubank", "klarna",
    "affirm", "afterpay", "uber", "lyft", "doordash", "ubereats", "grubhub", "instacart", "airbnb",
    "bookingdotcom", "expedia", "tripadvisor", "kayak", "delta", "unitedairlines", "americanairlines",
    "southwestairlines", "emirates", "qatarairways", "lufthansa", "britishairways", "ryanair",
    "spacex", "nasa", "esa", "blueorigin", "boeing", "airbus", "lockheedmartin", "raytheon",
    "generalmotors", "ford", "toyota", "honda", "bmw", "mercedes", "audi", "volkswagen", "porsche",
    "ferrari", "lamborghini", "mclaren", "bugatti", "kia", "hyundai", "nissan", "mazda", "subaru",
    "volvo", "rivian", "lucid", "nio", "byd", "polestar", "skoda", "seat", "jeep", "landrover",
    "jaguar", "astonmartin", "bentley", "rollsroyce", "maserati", "alfaromeo", "fiat", "peugeot",
    "renault", "citroen", "opel", "mini", "smart", "dacia", "geely", "greatwall", "chery",
    "samsung", "lg", "sony", "panasonic", "philips", "siemens", "bosch", "honeywell", "3m",
    "ge", "hpe", "dell", "lenovo", "asus", "acer", "msi", "razer", "logitech", "corsair",
    "steelseries", "hyperx", "roccat", "jabra", "bose", "jbl", "sonos", "beats", "shazam",
    "soundcloud", "deezer", "tidal", "pandora", "audible", "kindle", "kobo", "goodreads",
    "imdb", "rottentomatoes", "letterboxd", "crunchyroll", "disneyplus", "hulu", "hbomax", "primevideo",
    "peacock", "paramountplus", "appletv", "plex", "vlc", "obsstudio", "audacity", "davinciresolve",
    "finalcutpro", "logicpro", "ableton", "flstudio", "cubase", "protools", "rekordbox", "serato",
    "garmin", "fitbit", "whoop", "strava", "nike", "adidas", "puma", "underarmour", "reebok",
    "newbalance", "asics", "lululemon", "patagonia", "northface", "columbia", "decathlon",
    "ikea", "wayfair", "homedepot", "lowes", "costco", "walmart", "target", "alibaba", "aliexpress",
    "ebay", "etsy", "mercari", "poshmark", "depop", "shein", "temu", "wish", "flipkart", "meesho",
    "cocacola", "pepsi", "starbucks", "mcdonalds", "burgerking", "kfc", "subway", "dominos",
    "papajohns", "chipotle", "tacobell", "wendys", "dunkin", "redbull", "monster", "gatorade",
    "nike", "gucci", "prada", "chanel", "louisvuitton", "hermes", "dior", "versace", "armani",
    "burberry", "fendi", "givenchy", "balenciaga", "zara", "hm", "uniqlo", "gap", "levis",
    "thenorthface", "columbia", "patagonia", "thenorthface", "columbia", "patagonia",
    "harvard", "mit", "stanford", "yale", "princeton", "columbiauniversity", "berkeley", "caltech",
    "oxford", "cambridge", "ethzurich", "epfl", "coursera", "udemy", "edx", "khanacademy",
    "duolingo", "codecademy", "freecodecamp", "leetcode", "hackerrank", "codewars", "exercism",
    "stackoverflow", "stackexchange", "devdotto", "medium", "substack", "ghost", "wordpress",
    "blogger", "tumblr", "wikipedia", "wikimedia", "archiveofourown", "goodreads", "ycombinator",
    "producthunt", "indiegogo", "kickstarter", "patreon", "ko-fi", "buymeacoffee", "gumroad",
    "onlyfans", "substack", "beehiiv", "convertkit", "mailchimp", "sendgrid", "mailgun", "postmark",
    "hubspot", "salesforce", "zendesk", "intercom", "freshdesk", "helpscout", "drift", "crisp",
    "calendly", "doodle", "when2meet", "eventbrite", "meetup", "luma", "typeform", "surveymonkey",
    "googleforms", "qualtrics", "hotjar", "mixpanel", "amplitude", "segment", "heap", "plausible",
    "matomo", "googleanalytics", "adobeanalytics", "optimizely", "vwo", "launchdarkly", "flagsmith",
    "auth0", "okta", "onelogin", "duo", "lastpass", "1password", "bitwarden", "dashlane", "keeper",
    "protonmail", "protonvpn", "nordvpn", "expressvpn", "surfshark", "mullvad", "tailscale",
    "wireguard", "openvpn", "letsencrypt", "godaddy", "namecheap", "cloudflare", "dynatrace",
    "fastly", "akamai", "bunnydotnet", "imgix", "cloudinary", "uploadcare", "sanity", "contentful",
    "strapi", "directus", "payloadcms", "ghost", "keystone", "prismic", "storyblok", "builder",
    "webassembly", "wasm", "llvm", "gcc", "clang", "cmake", "ninja", "make", "gnu", "gnome",
    "kde", "xfce", "vim", "neovim", "emacs", "vscode", "visualstudio", "intellijidea", "pycharm",
    "webstorm", "goland", "rider", "clion", "datagrip", "androidstudio", "xcode", "eclipse",
    "netbeans", "sublimetext", "atom", "zed", "cursor", "windsurf", "copilot", "tabnine", "codeium",
    "replit", "codesandbox", "stackblitz", "gitpod", "coder", "jupyter", "colab", "kaggle",
    "databricks", "huggingface", "wandb", "mlflow", "kubeflow", "ray", "dvc", "weightsandbiases",
    "roboflow", "labelbox", "scale", "clarifai", "replicate", "stability", "midjourney", "runway",
    "elevenlabs", "anthropic", "cohere", "mistral", "perplexity", "groq", "cerebras", "together",
    "replicate", "modal", "banana", "baseten", "anyscale", "deepmind", "openai", "characterai",
    "poe", "pi", "claude", "gemini", "bard", "copilot", "siri", "alexa", "googleassistant",
    "nest", "ring", "arlo", "wyze", "ecobee", "honeywell", "philipshue", "lifx", "nanoleaf",
    "sonos", "homeassistant", "homebridge", "openhab", "tuya", "smartthings", "ifttt", "zapier",
    "make", "n8n", "pipedream", "trayio", "workato", "mulesoft", "boomi", "talend", "fivetran",
    "stitch", "airbyte", "hevo", "rivery", "matillion", "snaplogic", "informatica", "talend",
    "palantir", "snowflake", "databricks", "cloudera", "hortonworks", "mapr", "confluent",
    "redhat", "openshift", "vmware", "citrix", "nutanix", "proxmox", "virtualbox", "vmware",
    "qemu", "kvm", "xen", "hyperv", "parallels", "vagrant", "packer", "ansible", "chef", "puppet",
    "saltstack", "spinnaker", "argo", "flux", "helm", "kustomize", "skaffold", "tilt", "garden",
    "earthly", "buildkite", "teamcity", "bamboo", "octopusdeploy", "ansible", "pulumi", "crossplane",
]


def tint_svg(svg: str) -> str:
    svg = re.sub(r'fill="currentColor"', f'fill="{ICON_COLOR}"', svg, flags=re.I)
    svg = re.sub(r'stroke="currentColor"', f'stroke="{ICON_COLOR}"', svg, flags=re.I)
    svg = re.sub(r'fill="#[0-9A-Fa-f]{3,8}"', f'fill="{ICON_COLOR}"', svg)
    if ICON_COLOR not in svg[:240]:
        svg = svg.replace("<svg ", f'<svg fill="{ICON_COLOR}" ', 1)
    return svg


def download_icon(slug: str) -> tuple[str, bool, str]:
    path = os.path.join(ICON_DIR, f"{slug}.svg")
    if os.path.isfile(path) and os.path.getsize(path) > 40:
        return slug, True, "cached"
    url = f"{SVG_BASE}/{slug}.svg"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            svg = tint_svg(resp.read().decode("utf-8"))
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        return slug, True, "downloaded"
    except Exception as exc:
        if os.path.isfile(path):
            os.remove(path)
        return slug, False, str(exc)


def build_slug_list(all_slugs: set[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for slug in PRIORITY:
        if slug in all_slugs and slug not in seen:
            seen.add(slug)
            result.append(slug)

    for slug in sorted(all_slugs):
        if slug not in seen:
            seen.add(slug)
            result.append(slug)
        if len(result) >= TARGET_COUNT:
            break

    return result[:TARGET_COUNT]


def main() -> None:
    os.makedirs(ICON_DIR, exist_ok=True)

    print("Fetching simple-icons index...")
    with urllib.request.urlopen(INDEX_URL, timeout=30) as resp:
        index = json.loads(resp.read().decode("utf-8"))
    all_slugs = {
        os.path.splitext(os.path.basename(item["name"]))[0]
        for item in index.get("files", [])
        if item["name"].startswith("/icons/") and item["name"].endswith(".svg")
    }
    slugs = build_slug_list(all_slugs)
    print(f"Selected {len(slugs)} icon slugs")

    failed: list[str] = []
    ok = 0
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(download_icon, slug): slug for slug in slugs}
        for i, future in enumerate(as_completed(futures), 1):
            slug, success, msg = future.result()
            if success:
                ok += 1
            else:
                failed.append(slug)
                print(f"  FAIL {slug}: {msg}")
            if i % 100 == 0:
                print(f"  progress {i}/{len(slugs)}")

    # Replace failed slugs with extras from the catalog
    if failed:
        extras = [s for s in sorted(all_slugs) if s not in slugs]
        for slug in failed:
            while extras:
                alt = extras.pop(0)
                _, success, _ = download_icon(alt)
                if success and alt not in slugs:
                    slugs[slugs.index(slug)] = alt
                    ok += 1
                    break

    manifest = [s for s in slugs if os.path.isfile(os.path.join(ICON_DIR, f"{s}.svg"))]
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Done: {len(manifest)} icons in manifest ({ok} downloaded/cached)")


if __name__ == "__main__":
    main()