import asyncio
from pyppeteer import launch
import random
import numpy as np


# Assuming state is a string
state = "new_tab"

# Define the state space
states = [
    "new_tab",
    "open_google_docs",
    "name_document",
    "search_for_article",
    "click_on_article",
    "copy_paragraph",
    "paste_document",
    "change_font",
    "change_font_size",
    "finish_writing"
]

# Define the action space
actions = {
    "new_tab": "open_new_tab",
    "open_google_docs": "open_google_docs",
    "name_document": "name_document",
    "search_for_article": "search_google",
    "click_on_article": "click_result",
    "copy_paragraph": "copy_paragraph",
    "paste_document": "paste_document",
    "change_font": "change_font",
    "change_font_size": "change_font_size",
    "finish_writing": "finish_writing"
}

# Define the rewards
rewards = {
    "new_tab": 0.1,
    "open_google_docs": 0.1,
    "name_document": 0.1,
    "search_for_article": 0.1,
    "click_on_article": 0.1,
    "copy_paragraph": 0.1,
    "paste_document": 0.1,
    "change_font": 0.1,
    "change_font_size": 0.1,
    "finish_writing": 1.0,
    "invalid_action": -0.1
}


# Convert the state to a one-hot encoded vector
state_vector = np.zeros(len(states))
state_index = states.index(state)
state_vector[state_index] = 1


# Define the state size and action size
state_size = len(states)
action_size = len(actions)


# Define the task-specific functions

async def open_new_tab(browser):
    page = await browser.newPage()
    await page.goto('about:blank')
    return page


async def open_google_docs():
    browser = await launch(headless=False)
    page = await browser.newPage()
    await page.goto('https://docs.google.com/')
    return browser, page


async def name_document(page):
    await page.waitForSelector("div[aria-label='Docs']")
    docs_button = await page.querySelector("div[aria-label='Docs']")
    await docs_button.click()
    await page.waitForSelector("input[aria-label='Untitled document']")
    document_title = await page.querySelector("input[aria-label='Untitled document']")
    await document_title.type("machine learning")
    await page.keyboard.press("Enter")
    await page.waitForNavigation()


async def search_google(page):
    await page.waitForSelector("input[name='q']")
    search_input = await page.querySelector("input[name='q']")
    await search_input.type("machine learning")
    await search_input.press('Enter')


async def click_result(page):
    await page.waitForSelector("a[href*='http']")
    results = await page.querySelectorAll("a[href*='http']")
    random_result = random.choice(results)
    await random_result.click()


async def copy_paragraph(page):
    await page.waitForXPath("//p[contains(text(),'Machine learning is an application of artificial')]")
    paragraph = await page.xpath("//p[contains(text(),'Machine learning is an application of artificial')]")
    paragraph_text = await page.evaluate('(element) => element.textContent', paragraph[0])
    return paragraph_text


async def paste_document(page, text):
    await page.focus('body')
    await page.keyboard.type(text)

async def change_font(page):
    await page.waitForSelector("span[aria-label='Font family']")
    font_button = await page.querySelector("span[aria-label='Font family']")
    await font_button.click()
    await page.waitForSelector("div[aria-label='Times New Roman']")
    font = await page.querySelector("div[aria-label='Times New Roman']")
    await font.click()

async def change_font_size(page):
    await page.waitForSelector("span[aria-label='Font size']")
    size_button = await page.querySelector("span[aria-label='Font size']")
    await size_button.click()
    await page.waitForSelector("div[aria-label='12']")
    size = await page.querySelector("div[aria-label='12']")
    await size.click()

async def finish_writing(page):
    await page.waitForSelector("div[aria-label='File']")
    file_button = await page.querySelector("div[aria-label='File']")
    await file_button.click()
    await page.waitForSelector("div[aria-label='Close']")
    close_button = await page.querySelector("div[aria-label='Close']")
    await close_button.click()

def get_state(step, action):
    if step is None:
        return None  # Handle the case when step is None
    return f"{step}:{action}"

def get_reward(action):
    return rewards.get(action, rewards['invalid_action'])

async def perform_action(page, action):
    next_state = None
    reward = get_reward(action)

    if action == "new_tab":
        page = await actions[action](page)
        next_state = get_state("open_google_docs", None)
    elif action == "open_google_docs":
        browser, page = await actions[action]()
        next_state = get_state("name_document", None)
    elif action == "name_document":
        await actions[action](page)
        next_state = get_state("search_for_article", None)
    elif action == "search_for_article":
        await actions[action](page)
        next_state = get_state("click_on_article", None)
    elif action == "click_on_article":
        await actions[action](page)
        next_state = get_state("copy_paragraph", None)
    elif action == "copy_paragraph":
        paragraph = await actions[action](page)
        next_state = get_state("paste_document", paragraph)
    elif action == "paste_document":
        await actions[action](page, action.split(":")[1])
        next_state = get_state("change_font", None)
    elif action == "change_font":
        await actions[action](page)
        next_state = get_state("change_font_size", None)
    elif action == "change_font_size":
        await actions[action](page)
        next_state = get_state("finish_writing", None)
    elif action == "finish_writing":
        await actions[action](page)
        next_state = get_state("finish_writing", None)

    reward = get_reward(action)

    return next_state, reward



