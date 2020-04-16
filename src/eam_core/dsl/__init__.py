import logging
import math
import numbers
from distutils.util import strtobool

from antlr4 import InputStream, CommonTokenStream
from pint.quantity import _Quantity

from eam_core.dsl import MuParser
from eam_core.dsl.MuLexer import MuLexer
from eam_core.dsl.MuParser import MuParser
from eam_core.dsl.MuVisitor import MuVisitor

logger = logging.getLogger(__name__)


class EvalVisitor(MuVisitor):

    def __init__(self, variables=None, log_level=logging.WARNING, value_generator=None, debug=False):
        # CRITICAL = 50
        # FATAL = CRITICAL
        # ERROR = 40
        # WARNING = 30
        # WARN = WARNING
        # INFO = 20
        # DEBUG = 10
        # NOTSET = 0
        self.value_generator = value_generator
        self.level = log_level
        self.variables = variables
        self.result = None

    def visitAssignment(self, ctx):
        id = ctx.ID().getText()

        value = self.visit(ctx.expr())
        self.variables[id] = value
        return value

    def visitIdAtom(self, ctx):
        name = ctx.getText()
        if name in self.variables:
            return self.variables[name]
        else:
            raise Exception(
                "no such variable: " + name + " at line " + str(ctx.start.line) + " column " + str(ctx.start.column))

    def visitNumberAtom(self, ctx):
        value = float(ctx.getText())
        if self.value_generator is not None:
            value = self.value_generator(value)

        return value

    def visitStringAtom(self, ctx):
        return ctx.getText().strip("\"")

    def visitBooleanAtom(self, ctx: MuParser.BooleanAtomContext):
        return strtobool(ctx.getText())

    def visitNilAtom(self, ctx):
        return None

    def visitParExpr(self, ctx):
        return self.visit(ctx.expr())

    def visitPowExpr(self, ctx):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))

        if (isinstance(left, numbers.Real) and isinstance(right, numbers.Real)):
            return math.pow(left, right)

        import pandas
        if isinstance(left, pandas.DataFrame) or isinstance(left, pandas.Series):
            return left.pow(right)

    def visitUnaryMinusExpr(self, ctx: MuParser.UnaryMinusExprContext):
        # @todo - test
        value = self.visit(ctx.expr())
        return -value

    def visitNotExpr(self, ctx: MuParser.NotExprContext):
        # @todo - test
        value = self.visit(ctx.expr())
        return not value

    def visitMultiplicationExpr(self, ctx: MuParser.MultiplicationExprContext):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))

        _optype = ctx.op.type

        if _optype == MuParser.MULT:
            return left * right
        if _optype == MuParser.DIV:
            return left / right
        if _optype == MuParser.MOD:
            return left % right

        raise Exception("unknown operator: " + MuParser.tokenNames[ctx.op.type])

    def visitAdditiveExpr(self, ctx: MuParser.AdditiveExprContext):

        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))

        _optype = ctx.op.type
        if _optype == MuParser.PLUS:
            return left + right
        if _optype == MuParser.MINUS:
            return left - right

        raise Exception("unknown operator: " + MuParser.tokenNames[ctx.op.type])

    def visitRelationalExpr(self, ctx: MuParser.RelationalExprContext):

        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))

        _optype = ctx.op.type

        import pandas
        if isinstance(left, pandas.DataFrame) or isinstance(left, pandas.Series)\
                or isinstance(right, pandas.DataFrame) or isinstance(right, pandas.Series):
            if _optype == MuParser.LT:
                return (left < right).all().bool()
            if _optype == MuParser.LTEQ:
                return (left <= right).all().bool()
            if _optype == MuParser.GT:
                return (left > right).all().bool()
            if _optype == MuParser.GTEQ:
                return (left >= right).all().bool()
        else:
            if _optype == MuParser.LT:
                return left < right
            if _optype == MuParser.LTEQ:
                return left <= right
            if _optype == MuParser.GT:
                return left > right
            if _optype == MuParser.GTEQ:
                return left >= right

        raise Exception("unknown operator: " + MuParser.tokenNames[ctx.op.type])

    def visitEqualityExpr(self, ctx: MuParser.EqualityExprContext):
        # @todo test
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))

        _optype = ctx.op.type
        if isinstance(left, _Quantity):
            left = left.m
        if isinstance(right, _Quantity):
            right = right.m

        import pandas
        if isinstance(left, pandas.DataFrame) or isinstance(right, pandas.DataFrame):
            if _optype == MuParser.EQ:
                return (left == right).all().bool()
            if _optype == MuParser.NEQ:
                return (left != right).all().bool()
        if isinstance(left, pandas.Series) or isinstance(right, pandas.Series):
            if _optype == MuParser.EQ:
                return (left == right).all()
            if _optype == MuParser.NEQ:
                return (left != right).all()
        else:
            if _optype == MuParser.EQ:
                return left == right
            if _optype == MuParser.NEQ:
                return left != right

        raise Exception("unknown operator: " + MuParser.tokenNames[ctx.op.type])

    def visitAndExpr(self, ctx):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))

        if (isinstance(left, numbers.Real) or isinstance(left, bool)) and (
                isinstance(right, numbers.Real) or isinstance(right, bool)):
            return left and right
        else:
            import pandas
            if isinstance(left, pandas.DataFrame) or isinstance(left, pandas.Series):
                left = left.all().bool()
            if isinstance(right, pandas.DataFrame) or isinstance(right, pandas.Series):
                right = right.all().bool()
            return left and right

    def visitOrExpr(self, ctx):
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))

        if (isinstance(left, numbers.Real) or isinstance(left, bool)) and (
                isinstance(right, numbers.Real) or isinstance(right, bool)):
            return left or right

        else:
            import pandas
            if isinstance(left, pandas.DataFrame):
                left = left.all().bool()
            if isinstance(right, pandas.DataFrame):
                right = right.all().bool()

            return left or right

    def visitLog(self, ctx):
        left = self.visit(ctx.expr())
        logger.log(self.level, left)

    def visitIf_stat(self, ctx: MuParser.If_statContext):
        conditions = ctx.condition_block()
        evaluatedBlock = False
        condition: MuParser.Condition_blockContext
        for condition in conditions:
            evaluated = self.visit(condition.expr())

            if isinstance(evaluated, numbers.Real) or isinstance(evaluated, bool):
                if evaluated:
                    evaluatedBlock = True
                    self.visit(condition.stat_block())
                    break

            else:
                import pandas
                if isinstance(evaluated, pandas.DataFrame) or isinstance(evaluated, pandas.Series):
                    if evaluated.all().bool():
                        evaluatedBlock = True
                        self.visit(condition.stat_block())
                        break

        if not evaluatedBlock and ctx.stat_block() is not None:
            #  evaluate the else -stat_block ( if present == not null)
            self.visit(ctx.stat_block())

        return None

    def visitWhile_stat(self, ctx: MuParser.While_statContext):

        value = self.visit(ctx.expr())

        while value:
            # // evaluate the code block
            self.visit(ctx.stat_block())

            # // evaluate the expression
            value = self.visit(ctx.expr())

        return None

    def visitReturn_expr(self, ctx: MuParser.Return_exprContext):
        value = self.visit(ctx.expr())
        self.result = value
        return value


def evaluate(block, visitor=None, **kwargs):
    lexer = MuLexer(InputStream(block))
    stream = CommonTokenStream(lexer)
    parser = MuParser(stream)
    tree = parser.parse()
    if visitor == None:
        visitor = EvalVisitor(variables=kwargs.get('variables', {}))
    answer = visitor.visit(tree)
    # print(visitor.memory['a'])
    return visitor
