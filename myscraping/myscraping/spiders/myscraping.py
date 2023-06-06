import scrapy


class omyscraping(scrapy.Spider):
    name = "ourfirstbot"
    #allowed_domains = ["www.yelp.com"]
    #allowed_domains = ['www.amazon.com']
    #start_urls = ["https://www.yelp.com/search?find_desc=Restaurants"]
    #start_urls = ['https://www.amazon.com/-/zh/dp/B0B1M7S3XQ?ref_=Oct_DLandingS_D_dc8c8f6f_6&th=1']
    #start_urls = ['view-source:https://www.amazon.com/dp/B01MCYXPDB/ref=pd_rhf_d_cr_s_pd_sbs_rvi_sccl_1_1/140-5890132-8236045?pd_rd_w=11AGD&content-id=amzn1.sym.a089f039-4dde-401a-9041-8b534ae99e65&pf_rd_p=a089f039-4dde-401a-9041-8b534ae99e65&pf_rd_r=7XXRFVPB2D7K5MEYCZ73&pd_rd_wg=T32Me&pd_rd_r=970cc4c0-a360-4f34-a8be-e2ac6c0ee2e6&pd_rd_i=B01MCYXPDB&psc=1']
    allowed_domains = ['www.yelp.com/search?find_desc=Restaurants&find_loc=San Francisco, CA&ns=1']
    start_urls = ['https://www.yelp.com/search?find_desc=Restaurants&find_loc=San Francisco, CA&ns=1']



    def parse(self, response):
        print("%s : %s : %s" % (response.status, response.url, response.text))
        # yield response
        names = response.css('.heading--h3__373c0__1n4Of').extract()
        reviews = response.css('.reviewCount__373c0__2r4xT::text').extract()
        
        input()
        
        print('yaoyuan')
        # Give the extracted content row wise
        for item in zip(names, reviews):
            # create a dictionary to store the scraped info
            all_items = {
                'name': BeautifulSoup(item[0]).text,
                'reviews': item[1],

            }

            # yield or give the scraped info to scrapy
            yield all_items
