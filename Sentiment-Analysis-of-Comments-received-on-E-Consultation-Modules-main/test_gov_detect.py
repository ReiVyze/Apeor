from scraper.fetch_comments import is_government_site

def test_gov_detection():
    test_urls = {
        "https://regulations.gov": True,
        "https://mygov.nic.in": True,
        "https://ec.europa.eu/feedback": True,
        "https://www.google.com": False,
        "https://amazon.com/reviews": False
    }
    
    for url, expected in test_urls.items():
        result = is_government_site(url)
        print(f"URL: {url} | Expected: {expected} | Result: {result}")
        assert result == expected, f"Failed for {url}"

if __name__ == "__main__":
    try:
        test_gov_detection()
        print("\n✅ Gov Detection Logic verified successfully!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
