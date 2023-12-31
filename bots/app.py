from bot_one import BotOne
from bot_three import BotThree
import psycopg2
import time
import os
import pandas as pd
from collections import OrderedDict
from queue import Queue
from threading import Thread
from dotenv import load_dotenv
import json
import requests
load_dotenv()
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_USER = os.getenv("DATABASE_USER")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_ROOT_NAME = os.getenv("DATABASE_ROOT_NAME")

conn = psycopg2.connect(
    host=DATABASE_HOST,
    database=DATABASE_ROOT_NAME,
    user=DATABASE_USER,
    password=DATABASE_PASSWORD
)
cur = conn.cursor()


def sell_all_holdings():
    cur.execute(f"""
        SELECT uid, cost FROM portfolio;
    """)
    results = list(cur.fetchall())
    for i in range(len(results)):
        cur.execute(f"""
            UPDATE users SET cashvalue = (cashvalue+{round(results[i][1], 2)})
            WHERE uid='{results[i][0]}';
        """)
        conn.commit()


def cancel_all_orders():
    cur.execute(f"""
        SELECT order_id, user_uid, price, shares, action, company_name FROM orders 
        WHERE accepted={False};
    """)
    results = list(cur.fetchall())
    for result in results:
        order_id = result[0]
        user_uid = result[1]
        price = result[2]
        shares = result[3]
        action = result[4]
        company_name = result[5]
        trade_value = price * shares
        cur.execute(f"""
            DELETE FROM orders WHERE order_id='{order_id}';
        """)
        conn.commit()
        if action == "buy":
            cur.execute(f"""
                UPDATE users SET cashvalue = (cashvalue+{round(trade_value, 2)})
                WHERE uid='{user_uid}';
            """)
            conn.commit()
        elif action == "sell":
            cur.execute(f"""
                UPDATE portfolio SET pending_shares_holding = (pending_shares_holding+{round(shares,2)})
                WHERE uid='{user_uid}' and company_id='{company_name}';
            """)
            conn.commit()


def automated_cancel_order(time_difference=60*60*24):
    current_time = time.time()
    cur.execute(f"""
        SELECT order_id, user_uid, price, shares, action, company_name FROM orders 
        WHERE {current_time}-timestamp > {time_difference} AND accepted={False};
    """)
    results = list(cur.fetchall())
    for result in results:
        order_id = result[0]
        user_uid = result[1]
        price = result[2]
        shares = result[3]
        action = result[4]
        company_name = result[5]
        trade_value = price * shares
        cur.execute(f"""
            DELETE FROM orders WHERE order_id='{order_id}';
        """)
        conn.commit()
        if action == "buy":
            cur.execute(f"""
                UPDATE users SET cashvalue = (cashvalue+{round(trade_value, 2)})
                WHERE uid='{user_uid}';
            """)
            conn.commit()
        elif action == "sell":
            cur.execute(f"""
                UPDATE portfolio SET pending_shares_holding = (pending_shares_holding+{round(shares,2)})
                WHERE uid='{user_uid}' and company_id='{company_name}';
            """)
            conn.commit()


def automated_cancel_bot_order(time_difference=18):
    current_time = time.time()
    cur.execute(f"""
        SELECT order_id, bot_id, price, shares, action, company_name FROM bot_orders 
        WHERE {current_time}-timestamp > {time_difference} AND accepted={False};
    """)
    results = list(cur.fetchall())
    if len(results) != 0:
        print(f"canceling {len(results)} results")
        for result in results:
            order_id = result[0]
            bot_id = result[1]
            price = result[2]
            shares = result[3]
            action = result[4]
            company_name = result[5]
            trade_value = price * shares
            cur.execute(f"""
                DELETE FROM bot_orders WHERE order_id='{order_id}';
            """)
            conn.commit()
            if action == "buy":
                cur.execute(f"""
                    UPDATE bots SET cashvalue = (cashvalue+{round(trade_value, 2)})
                    WHERE bot_id='{bot_id}';
                """)
                conn.commit()
            elif action == "sell":
                cur.execute(f"""
                    UPDATE bot_portfolio SET pending_shares_holding = (pending_shares_holding+{round(shares,2)})
                    WHERE bot_id='{bot_id}' and company_id='{company_name}';
                """)
                conn.commit()


def get_price_from_database(company_id):
    cur.execute(f"""
          SELECT price_list from prices WHERE company_id='{company_id}';
    """)
    price = list(cur.fetchone()[0])
    price = [float(i) for i in price]
    print(price[0])
    return price


company_lst = ["ast", "dsc", "fsin", "hhw", "jky", "sgo", "wrkn"]


def register_bot(register_url, bot_name, initial_price):
    payload = json.dumps({
        "bot_name": bot_name,
        "initial_price": initial_price
    })
    response = requests.request("POST", register_url, data=payload)
    print(response.status_code)


def get_active_order_book(comp_name):
    # get all buy orders from ranked from high to low
    cur.execute(f"""
        SELECT price, shares, RANK() OVER (ORDER BY price DESC) as rank FROM orders WHERE 
        accepted={False} AND company_name='{comp_name}' AND action='buy';
    """)
    buy_orders = list(cur.fetchall())
    buy_order_book = OrderedDict(())
    if buy_orders != []:
        price = buy_orders[0][0]
        shares = buy_orders[0][1]
        for index in range(1, len(buy_orders)):
            if buy_orders[index][0] == price:
                shares += buy_orders[index][1]
            else:
                buy_order_book.update({float(price): -float(shares)})
                price = buy_orders[index][0]
                shares = buy_orders[index][1]

            if index + 1 == len(buy_orders):
                buy_order_book.update({float(price): -float(shares)})
        if len(buy_orders) == 1:
            buy_order_book.update({float(price): -float(shares)})
    cur.execute(f"""
        SELECT price, shares, RANK() OVER (ORDER BY price DESC) as rank FROM orders WHERE 
        accepted={False} AND company_name='{comp_name}' AND action='sell';
    """)
    sell_orders = list(cur.fetchall())
    sell_order_book = OrderedDict(())
    if sell_orders != []:
        price = sell_orders[0][0]
        shares = sell_orders[0][1]
        for index in range(1, len(sell_orders)):
            if sell_orders[index][0] == price:
                shares += sell_orders[index][1]
            else:
                sell_order_book.update({float(price): float(shares)})
                price = sell_orders[index][0]
                shares = sell_orders[index][1]

            if index + 1 == len(sell_orders):
                sell_order_book.update({float(price): float(shares)})
        if len(sell_orders) == 1:
            sell_order_book.update({float(price): float(shares)})
    sell_order_book.update(buy_order_book)
    if sell_order_book == OrderedDict(()):
        sell_order_book.update({0: 0})
    return sell_order_book


def trader(result_queue, price_info, time_stamp, shares=10):
    trade = {"trader bot type": "enters trader decisions"}
    result_queue.put(["trader", trade])

def accepter(result_queue, price_info, time_stamp, order_book):
    accepter = {"accepter bot type": "enters accepter decisions"}
    result_queue.put(["accepter", accepter])


def bidder(result_queue, price_info, time_stamp, shares=50, split=50):
    bidder = {"bidder bot type": "enters bidder decisions"}
    result_queue.put(["bidder", bidder])


if __name__ == '__main__':
    start_time = time.time() - 60*60*24*10 - 60*60*10

    bot1 = BotOne()
    bot2 = BotThree()

    ast_price = get_price_from_database("ast")
    ast_price_df = pd.DataFrame(ast_price, columns=["price"])["price"]
    dsc_price = get_price_from_database("dsc")
    dsc_price_df = pd.DataFrame(dsc_price, columns=["price"])["price"]
    fsin_price = get_price_from_database("fsin")
    fsin_price_df = pd.DataFrame(fsin_price, columns=["price"])["price"]
    hhw_price = get_price_from_database("hhw")
    hhw_price_df = pd.DataFrame(hhw_price, columns=["price"])["price"]
    jky_price = get_price_from_database("jky")
    jky_price_df = pd.DataFrame(jky_price, columns=["price"])["price"]
    sgo_price = get_price_from_database("sgo")
    sgo_price_df = pd.DataFrame(sgo_price, columns=["price"])["price"]
    wrkn_price = get_price_from_database("wrkn")
    wrkn_price_df = pd.DataFrame(wrkn_price, columns=["price"])["price"]

    price_info = {
        "ast": ast_price_df,
        "dsc": dsc_price_df,
        "fsin": fsin_price_df,
        "hhw": hhw_price_df,
        "jky": jky_price_df,
        "sgo": sgo_price_df,
        "wrkn": wrkn_price_df,
    }

    order_book = {
        "ast": get_active_order_book("ast"),
        "dsc": get_active_order_book("dsc"),
        "fsin": get_active_order_book("fsin"),
        "hhw": get_active_order_book("hhw"),
        "jky": get_active_order_book("jky"),
        "sgo": get_active_order_book("sgo"),
        "wrkn": get_active_order_book("wrkn"),
    }

    initial_price = {
        "ast": ast_price[0],
        "dsc": dsc_price[0],
        "fsin": fsin_price[0],
        "hhw": hhw_price[0],
        "jky": jky_price[0],
        "sgo": sgo_price[0],
        "wrkn": wrkn_price[0]
    }

    register_bot("[#request endpoint]",
                 "Bot1", initial_price)
    register_bot("[#request endpoint]",
                 "Bot2", initial_price)

    # initial_index = int(time.time()-start_time)
    index = int(time.time()-start_time)
    # each loop takes around 2.5 seconds

    while index <= len(ast_price):
        # automated_cancel_order()
        begin_time = time.time()
        bot_data = {}

        result_queue = Queue()
        time_stamp = index
        split = 50
        if time_stamp >= split:
            automated_cancel_bot_order()
            automated_cancel_order()
            t1 = Thread(target=trader, args=(
                result_queue, price_info, time_stamp))
            t2 = Thread(target=accepter, args=(
                result_queue, price_info, time_stamp, order_book))
            t3 = Thread(target=bidder, args=(
                result_queue, price_info, time_stamp))

            t1.start()
            t2.start()
            t3.start()

            t1.join()
            t2.join()
            t3.join()
        else:
            t1 = Thread(target=trader, args=(
                result_queue, price_info, time_stamp))
            t2 = Thread(target=accepter, args=(
                result_queue, price_info, time_stamp, order_book))

            t1.start()
            t2.start()

            t1.join()
            t2.join()

        while not result_queue.empty():
            result = result_queue.get()
            bot_data[result[0]] = result[1]

        if "bidder" not in bot_data:
            bot_data["bidder"] = {
                "index": {
                    #enter bot trade data
                }
            }
        print(f"requesting: {bot_data['bidder']}")

        response = requests.request(
            "POST", "[#request endpoint]", data=json.dumps(bot_data))
        index += int(time.time()-begin_time)
        time.sleep(1)

    cancel_all_orders()
    sell_all_holdings()
    print("orders canceled")