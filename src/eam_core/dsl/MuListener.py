# Generated from Mu.g4 by ANTLR 4.7.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .MuParser import MuParser
else:
    from eam_core.dsl.MuParser import MuParser

# This class defines a complete listener for a parse tree produced by MuParser.
class MuListener(ParseTreeListener):

    # Enter a parse tree produced by MuParser#parse.
    def enterParse(self, ctx:MuParser.ParseContext):
        pass

    # Exit a parse tree produced by MuParser#parse.
    def exitParse(self, ctx:MuParser.ParseContext):
        pass


    # Enter a parse tree produced by MuParser#block.
    def enterBlock(self, ctx:MuParser.BlockContext):
        pass

    # Exit a parse tree produced by MuParser#block.
    def exitBlock(self, ctx:MuParser.BlockContext):
        pass


    # Enter a parse tree produced by MuParser#stat.
    def enterStat(self, ctx:MuParser.StatContext):
        pass

    # Exit a parse tree produced by MuParser#stat.
    def exitStat(self, ctx:MuParser.StatContext):
        pass


    # Enter a parse tree produced by MuParser#return_expr.
    def enterReturn_expr(self, ctx:MuParser.Return_exprContext):
        pass

    # Exit a parse tree produced by MuParser#return_expr.
    def exitReturn_expr(self, ctx:MuParser.Return_exprContext):
        pass


    # Enter a parse tree produced by MuParser#assignment.
    def enterAssignment(self, ctx:MuParser.AssignmentContext):
        pass

    # Exit a parse tree produced by MuParser#assignment.
    def exitAssignment(self, ctx:MuParser.AssignmentContext):
        pass


    # Enter a parse tree produced by MuParser#if_stat.
    def enterIf_stat(self, ctx:MuParser.If_statContext):
        pass

    # Exit a parse tree produced by MuParser#if_stat.
    def exitIf_stat(self, ctx:MuParser.If_statContext):
        pass


    # Enter a parse tree produced by MuParser#condition_block.
    def enterCondition_block(self, ctx:MuParser.Condition_blockContext):
        pass

    # Exit a parse tree produced by MuParser#condition_block.
    def exitCondition_block(self, ctx:MuParser.Condition_blockContext):
        pass


    # Enter a parse tree produced by MuParser#stat_block.
    def enterStat_block(self, ctx:MuParser.Stat_blockContext):
        pass

    # Exit a parse tree produced by MuParser#stat_block.
    def exitStat_block(self, ctx:MuParser.Stat_blockContext):
        pass


    # Enter a parse tree produced by MuParser#while_stat.
    def enterWhile_stat(self, ctx:MuParser.While_statContext):
        pass

    # Exit a parse tree produced by MuParser#while_stat.
    def exitWhile_stat(self, ctx:MuParser.While_statContext):
        pass


    # Enter a parse tree produced by MuParser#log.
    def enterLog(self, ctx:MuParser.LogContext):
        pass

    # Exit a parse tree produced by MuParser#log.
    def exitLog(self, ctx:MuParser.LogContext):
        pass


    # Enter a parse tree produced by MuParser#notExpr.
    def enterNotExpr(self, ctx:MuParser.NotExprContext):
        pass

    # Exit a parse tree produced by MuParser#notExpr.
    def exitNotExpr(self, ctx:MuParser.NotExprContext):
        pass


    # Enter a parse tree produced by MuParser#unaryMinusExpr.
    def enterUnaryMinusExpr(self, ctx:MuParser.UnaryMinusExprContext):
        pass

    # Exit a parse tree produced by MuParser#unaryMinusExpr.
    def exitUnaryMinusExpr(self, ctx:MuParser.UnaryMinusExprContext):
        pass


    # Enter a parse tree produced by MuParser#multiplicationExpr.
    def enterMultiplicationExpr(self, ctx:MuParser.MultiplicationExprContext):
        pass

    # Exit a parse tree produced by MuParser#multiplicationExpr.
    def exitMultiplicationExpr(self, ctx:MuParser.MultiplicationExprContext):
        pass


    # Enter a parse tree produced by MuParser#atomExpr.
    def enterAtomExpr(self, ctx:MuParser.AtomExprContext):
        pass

    # Exit a parse tree produced by MuParser#atomExpr.
    def exitAtomExpr(self, ctx:MuParser.AtomExprContext):
        pass


    # Enter a parse tree produced by MuParser#orExpr.
    def enterOrExpr(self, ctx:MuParser.OrExprContext):
        pass

    # Exit a parse tree produced by MuParser#orExpr.
    def exitOrExpr(self, ctx:MuParser.OrExprContext):
        pass


    # Enter a parse tree produced by MuParser#additiveExpr.
    def enterAdditiveExpr(self, ctx:MuParser.AdditiveExprContext):
        pass

    # Exit a parse tree produced by MuParser#additiveExpr.
    def exitAdditiveExpr(self, ctx:MuParser.AdditiveExprContext):
        pass


    # Enter a parse tree produced by MuParser#powExpr.
    def enterPowExpr(self, ctx:MuParser.PowExprContext):
        pass

    # Exit a parse tree produced by MuParser#powExpr.
    def exitPowExpr(self, ctx:MuParser.PowExprContext):
        pass


    # Enter a parse tree produced by MuParser#relationalExpr.
    def enterRelationalExpr(self, ctx:MuParser.RelationalExprContext):
        pass

    # Exit a parse tree produced by MuParser#relationalExpr.
    def exitRelationalExpr(self, ctx:MuParser.RelationalExprContext):
        pass


    # Enter a parse tree produced by MuParser#equalityExpr.
    def enterEqualityExpr(self, ctx:MuParser.EqualityExprContext):
        pass

    # Exit a parse tree produced by MuParser#equalityExpr.
    def exitEqualityExpr(self, ctx:MuParser.EqualityExprContext):
        pass


    # Enter a parse tree produced by MuParser#andExpr.
    def enterAndExpr(self, ctx:MuParser.AndExprContext):
        pass

    # Exit a parse tree produced by MuParser#andExpr.
    def exitAndExpr(self, ctx:MuParser.AndExprContext):
        pass


    # Enter a parse tree produced by MuParser#parExpr.
    def enterParExpr(self, ctx:MuParser.ParExprContext):
        pass

    # Exit a parse tree produced by MuParser#parExpr.
    def exitParExpr(self, ctx:MuParser.ParExprContext):
        pass


    # Enter a parse tree produced by MuParser#numberAtom.
    def enterNumberAtom(self, ctx:MuParser.NumberAtomContext):
        pass

    # Exit a parse tree produced by MuParser#numberAtom.
    def exitNumberAtom(self, ctx:MuParser.NumberAtomContext):
        pass


    # Enter a parse tree produced by MuParser#booleanAtom.
    def enterBooleanAtom(self, ctx:MuParser.BooleanAtomContext):
        pass

    # Exit a parse tree produced by MuParser#booleanAtom.
    def exitBooleanAtom(self, ctx:MuParser.BooleanAtomContext):
        pass


    # Enter a parse tree produced by MuParser#idAtom.
    def enterIdAtom(self, ctx:MuParser.IdAtomContext):
        pass

    # Exit a parse tree produced by MuParser#idAtom.
    def exitIdAtom(self, ctx:MuParser.IdAtomContext):
        pass


    # Enter a parse tree produced by MuParser#stringAtom.
    def enterStringAtom(self, ctx:MuParser.StringAtomContext):
        pass

    # Exit a parse tree produced by MuParser#stringAtom.
    def exitStringAtom(self, ctx:MuParser.StringAtomContext):
        pass


    # Enter a parse tree produced by MuParser#nilAtom.
    def enterNilAtom(self, ctx:MuParser.NilAtomContext):
        pass

    # Exit a parse tree produced by MuParser#nilAtom.
    def exitNilAtom(self, ctx:MuParser.NilAtomContext):
        pass


