def removeDuplicateFuzzy():
    """ 
    Rule: 
        Normalize
        1. If records with exact same email/cid - merge immediately
        2. If Primary Phone Number matche exactly: check levenshtein distance for "Name" - similarity > 85% -> Merge
        3. If 'Name' matches exactly - check for email, phone similarity - if email similarity>80% or Phone matches exactly -> Merge 
    """

