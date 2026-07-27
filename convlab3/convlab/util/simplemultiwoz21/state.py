def default_state():
    state = dict(user_action=[],
                 system_action=[],
                 belief_state={
                     # 'hotel': {'name': '', 'area': '', 'parking': '', 'price range': '', 'stars': '4', 'internet': 'yes', 'type': 'hotel', 'book stay': '', 'book day': '', 'book people': ''}, 
                     'hotel': {'name': '', 'area': '', 'parking': '', 'price range': '', 'stars': '', 'internet': '', 'type': '', 'book stay': '', 'book day': '', 'book people': ''}, 
                     'restaurant': {'food': '', 'price range': '', 'name': '', 'area': '', 'book time': '', 'book day': '', 'book people': ''} 
                     },
                 booked={},
                 request_state={},
                 terminated=False,
                 history=[])
    return state


def default_state_old():
    state = dict(user_action=[],
                 system_action=[],
                 belief_state={},
                 request_state={},
                 terminated=False,
                 history=[])
    state['belief_state'] = {
        "hotel": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "",
                "pricerange": "",
                "name": "",
                "area": "",
            }
        }
    }
    return state
