from browser_use_sdk import BrowserUseSdk

sdk = BrowserUseSdk(api_key="your api key here")

task_performance = """
Please perform these steps in sequence on the Chanakya University website:

1. Open the homepage: https://chanakyauniversity.in/
2. Scroll down to and click on the 'Programs' or 'Acquire' menu (whichever is present).
3. From the dropdown, navigate to 'Undergraduate Programmes'.
4. On that page, click on 'School of Engineering'.
5. Then click on 'School of Arts, Humanities and Social Sciences'.
6. Finally, navigate to the 'About Chanakya University' page.

After each step, summarize the page title or key heading you’ve reached.
"""
result = sdk.run(
    llm_model="o3",
    task_performance=task_performance)