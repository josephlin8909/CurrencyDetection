import requests
import json


# Daniel Choi and Sohee Kim (12/4/22) - API call to convert currency to USD
def calc_exchange_rate(currency_in_type: int, currency_in_denom: int):
    # get currency the user is in
    if currency_in_type == 0:
        currency_in = "COP"
    elif currency_in_type == 1:
        currency_in = "THB"
    else:
        return 0
    money_in = currency_in_denom
    # get what currency the user wants
    currency_out = "USD"

    # convert
    url_part = 'https://api.exchangerate-api.com/v4/latest/'
    url_combined = url_part + currency_in
    response = requests.get(url_combined)
    data = json.loads(response.text)

    # print(type(data)) #dic

    conversions = data['rates'][str(currency_out)]
    money_out = money_in * conversions
    print("The ammount in " + currency_out + " is " + str(money_out))
    return round(money_out, 2)
