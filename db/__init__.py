import pymysql
from pymysql import cursors

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='123456',
                             database='autocomplete',
                             cursorclass=pymysql.cursors.DictCursor)
