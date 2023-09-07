"""
COMS W4705 - Natural Language Processing - Spring 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer

Code within all the 'COMPLETE THIS METHOD/TODO' sections was written by Xingchen (Estella) Ye.
"""

import sys
from collections import defaultdict
from math import fsum
from math import isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1

        s = self.lhs_to_rules.keys()

        for lhs in s:
            p = 0.0
            for tuple in self.lhs_to_rules[lhs]:
                if len(tuple[1]) == 2:
                    if tuple[1][0] not in s \
                    or tuple[1][1] not in s:
                        print("not CNF, not a non-terminal")
                        return False
                elif len(tuple[1]) == 1:
                    if tuple[1][0] in s:
                        print("not CNF, not terminal")
                        return False

                p += tuple[2]
            if not isclose(p, 1.0):
                print("not CNF, probability not sum up to 1.0")
                return False

        return True 


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
    
    grammar.verify_grammar()
        
