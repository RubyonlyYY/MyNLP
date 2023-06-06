# -*- coding: utf-8 -*-
import scrapy
from bs4 import BeautifulSoup


class MyfirstbotSpider(scrapy.Spider):
    name = 'myfirstbot'
    start_urls = [
        'https://www.yelp.com/search?find_desc=Restaurants&find_loc=San Francisco, CA&ns=1',
    ]

    def parse(self, response):
        # yield response
        names = response.css('.heading--h3__373c0__1n4Of').extract()
        reviews = response.css('.reviewCount__373c0__2r4xT::text').extract()

        # Give the extracted content row wise
        for item in zip(names, reviews):
            # create a dictionary to store the scraped info
            all_items = {
                'name': BeautifulSoup(item[0]).text,
                'reviews': item[1],

            }

            # yield or give the scraped info to scrapy
            yield all_items