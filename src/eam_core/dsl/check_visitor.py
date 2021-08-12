import logging
from collections import namedtuple

from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ConsoleErrorListener, ErrorListener

from eam_core.dsl import SimpleErrorThrower, MyErrorStrategy
from eam_core.dsl.MuLexer import MuLexer
from eam_core.dsl.MuParser import MuParser, BailErrorStrategy, Parser, RecognitionException
from eam_core.dsl.MuVisitor import MuVisitor

logger = logging.getLogger(__name__)

BranchVars = namedtuple('BranchVars', ['new_variables', 'implicit_variables'])


class CheckVisitor(MuVisitor):
    """
    performs static analysis.

    - parses the DSL and identifies all variables used (ID atoms)
    - finds all occurring variables
    - tracks assignments to new variables
    - recurs into all branches

    """

    def __init__(self, log_level=logging.WARNING, debug=False):
        # CRITICAL = 50
        # FATAL = CRITICAL
        # ERROR = 40
        # WARNING = 30
        # WARN = WARNING
        # INFO = 20
        # DEBUG = 10
        # NOTSET = 0

        self.level = log_level
        self.new_variables = set()
        self.implicit_variables = set()

    def visitAssignment(self, ctx):
        """
        an implicit variable should not be recorded as a new variable
        :param ctx:
        :return:
        """
        name = ctx.ID().getText()

        self.visit(ctx.expr())
        if name not in self.implicit_variables:
            logger.debug(
                "New variable " + name + " created at line " + str(ctx.start.line) + " column " + str(ctx.start.column))
            self.new_variables.add(name)

    def visitIdAtom(self, ctx):
        """
        an reference to a symbol that has not been created in the formula as considered an implicit variable
        :param ctx:
        :return:
        """
        name = ctx.getText()
        if name not in self.new_variables:
            logger.debug(
                "Implicit variable <" + name + "> referenced at line " + str(ctx.start.line) +
                " column " + str(ctx.start.column))
            self.implicit_variables.add(name)

    def visitIf_stat(self, ctx: MuParser.If_statContext):
        """
        visit all conditions and all stat blocks, including the else block.

        we want to track implicit assignments in any branch, independent of order

        we evaluate conditions and stats independently and compare variable references at the end of the if/else block

        we track **any** implicit variables, we track new variables only if they are not also an implicit variable

        Below, b is only tracked as an implicit variable, not a new variable
                   a = 0 ;
                   if (a != 0){
                        b = 2;
                   } else if (b != 0){
                        c = 1;
                   } else {
                        c = 2;
                   }

        :param ctx:
        :return:
        """
        conditions = ctx.condition_block()
        # store the implicit and new variables in each block
        branch_vars = []
        shelf = BranchVars(self.new_variables.copy(), self.implicit_variables.copy())

        condition: MuParser.Condition_blockContext
        for condition in conditions:
            # reset variable stack for condition
            self.new_variables = shelf.new_variables.copy()
            self.implicit_variables = shelf.implicit_variables.copy()

            # evaluate
            self.visit(condition.expr())
            self.visit(condition.stat_block())

            # store variables
            vars = BranchVars(self.new_variables.copy(), self.implicit_variables.copy())
            branch_vars.append(vars)

        if ctx.stat_block() is not None:
            #  evaluate the else -stat_block ( if present == not null)

            # reset variable stack for condition
            self.new_variables = shelf.new_variables.copy()
            self.implicit_variables = shelf.implicit_variables.copy()

            self.visit(ctx.stat_block())

            # store variables
            vars = BranchVars(self.new_variables.copy(), self.implicit_variables.copy())
            branch_vars.append(vars)

        # compare:
        # we want to track **any** implicit variables, we want to track all new variables
        # restore the variable state as before
        self.new_variables = shelf.new_variables
        self.implicit_variables = shelf.implicit_variables

        for _condition_variables in branch_vars:
            for _var in _condition_variables.implicit_variables:
                self.implicit_variables.add(_var)
        for _condition_variables in branch_vars:
            for _var in _condition_variables.new_variables:
                if _var not in self.implicit_variables:
                    self.new_variables.add(_var)

        return None

    def visitWhile_stat(self, ctx: MuParser.While_statContext):

        self.visit(ctx.expr())

        # // evaluate the code block
        self.visit(ctx.stat_block())

        return None

    # todo: untested
    def visitReturn_expr(self, ctx: MuParser.Return_exprContext):
        value = self.visit(ctx.expr())
        self.result = value
        return value


def evaluate(block, visitor=None, **kwargs):
    lexer = MuLexer(InputStream(block))
    stream = CommonTokenStream(lexer)
    parser = MuParser(stream)
    parser.removeErrorListeners()
    parser.addErrorListener(SimpleErrorThrower())
    parser._errHandler = MyErrorStrategy()
    tree = parser.parse()
    if visitor == None:
        visitor = CheckVisitor()
    visitor.visit(tree)
    return visitor
