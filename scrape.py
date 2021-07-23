import os
from selenium import webdriver
import time
from PIL import Image
import urllib
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.keys import Keys

#Install Driver
driver = webdriver.Edge(EdgeChromiumDriverManager().install())
driver.get('https://www.google.com/')
search = driver.find_element_by_name('q')
search.send_keys('Toyota Rav4',Keys.ENTER)

elem = driver.find_element_by_link_text('Images')
elem.get_attribute('href')
elem.click()

value = 0
for i in range(200): 
    driver.execute_script('scrollBy("+ str(value) +",+100);')
    value += 100
    time.sleep(4)

elements = driver.find_elements_by_xpath('//img[contains(@class,"rg_i")]')
count = 0
for i in elements:
    src = i.get_attribute('src')
    try:
        if src != None:
            src  = str(src)
            count+=1
            urllib.request.urlretrieve(src, os.path.join('images/toyota/rav4','rav4'+str(count)+'.jpg'))
        else:
            raise TypeError
    except TypeError:
        pass
