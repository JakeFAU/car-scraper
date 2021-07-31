import os
from selenium import webdriver
import time
from PIL import Image
import urllib
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.keys import Keys
import json
import shutil

#Install Driver
driver = webdriver.Edge(EdgeChromiumDriverManager().install())

#Parse JSON
with open('scrape2.json', 'r') as json_file:
  car_input = json.load(json_file)

cars = car_input["Toyota"]
for model in cars:
    for generations in cars[model]:
        generation = str(generations["generation"])
        path = os.path.join("images-generation",model, generation)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

        query = model + " " + generation
        driver.get('https://www.google.com/')
        search = driver.find_element_by_name('q')
        search.send_keys(query,Keys.ENTER)

        elem = driver.find_element_by_link_text('Images')
        elem.get_attribute('href')
        elem.click()

        value = 0
        for i in range(300): 
            driver.execute_script('scrollBy("+ str(value) +",+100);')
            value += 100
            time.sleep(.5)

        elements = driver.find_elements_by_xpath('//img[contains(@class,"rg_i")]')
        count = 0
        for i in elements:
            src = i.get_attribute('src')
            try:
                if src != None:
                    src  = str(src)
                    count+=1
                    urllib.request.urlretrieve(src, os.path.join(path,generation+"_"+str(count)+'.jpg'))
                else:
                    raise TypeError
            except Exception:
                pass


