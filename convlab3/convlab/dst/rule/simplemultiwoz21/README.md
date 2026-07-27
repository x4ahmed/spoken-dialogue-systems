# RuleDST
RuleDST is a rule based dialog state tracker which trivially updates new values from NLU result to states.

# How to use
Example:

```python
from convlab.dst.rule.simplemultiwoz21 import RuleDST


dst = RuleDST()

# Action is a dict. Its keys are strings(domain-type pairs, both uppercase and lowercase is OK) and its values are list of lists. 
# The domain may be one of ('Hotel', 'Restaurant').
# The type may be "inform" or "request".

# For example, the action below has a key "Hotel-Inform", in which "Hotel" is domain and "Inform" is action type.
# Each list in the value of "Hotel-Inform" is a slot-value pair. "Area" is slot and "east" is value. "Star" is slot and "4" is value.
action = [
    ["Inform", "Hotel", "Area", "east"],
    ["Inform", "Hotel", "Stars", "4"]
]

# method `update` updates the attribute `state` of tracker, and returns it.
state = dst.update(action)
assert state == dst.state
assert state == {'user_action': [["Inform", "Hotel", "Area", "east"], ["Inform", "Hotel", "Stars", "4"]],
 'system_action': [],
 'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                  'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                            'semi': {'name': '',
                                     'area': 'east',
                                     'parking': '',
                                     'pricerange': '',
                                     'stars': '4',
                                     'internet': '',
                                     'type': ''}},
                  'attraction': {'book': {'booked': []},
                                 'semi': {'type': '', 'name': '', 'area': ''}},
                  'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                 'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                  'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                  'taxi': {'book': {'booked': []},
                           'semi': {'leaveAt': '',
                                    'destination': '',
                                    'departure': '',
                                    'arriveBy': ''}},
                  'train': {'book': {'booked': [], 'people': ''},
                            'semi': {'leaveAt': '',
                                     'destination': '',
                                     'day': '',
                                     'arriveBy': '',
                                     'departure': ''}}},
 'request_state': {},
 'terminated': False,
 'history': []}
```
