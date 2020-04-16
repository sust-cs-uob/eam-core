# Generated from Mu.g4 by ANTLR 4.7.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .MuParser import MuParser
else:
    from eam_core.dsl.MuParser import MuParser

# This class defines a complete generic visitor for a parse tree produced by MuParser.

class MuVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by MuParser#parse.
    def visitParse(self, ctx:MuParser.ParseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#block.
    def visitBlock(self, ctx:MuParser.BlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#stat.
    def visitStat(self, ctx:MuParser.StatContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#return_expr.
    def visitReturn_expr(self, ctx:MuParser.Return_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#assignment.
    def visitAssignment(self, ctx:MuParser.AssignmentContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#if_stat.
    def visitIf_stat(self, ctx:MuParser.If_statContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#condition_block.
    def visitCondition_block(self, ctx:MuParser.Condition_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#stat_block.
    def visitStat_block(self, ctx:MuParser.Stat_blockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#while_stat.
    def visitWhile_stat(self, ctx:MuParser.While_statContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#log.
    def visitLog(self, ctx:MuParser.LogContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#notExpr.
    def visitNotExpr(self, ctx:MuParser.NotExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#unaryMinusExpr.
    def visitUnaryMinusExpr(self, ctx:MuParser.UnaryMinusExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#multiplicationExpr.
    def visitMultiplicationExpr(self, ctx:MuParser.MultiplicationExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#atomExpr.
    def visitAtomExpr(self, ctx:MuParser.AtomExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#orExpr.
    def visitOrExpr(self, ctx:MuParser.OrExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#additiveExpr.
    def visitAdditiveExpr(self, ctx:MuParser.AdditiveExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#powExpr.
    def visitPowExpr(self, ctx:MuParser.PowExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#relationalExpr.
    def visitRelationalExpr(self, ctx:MuParser.RelationalExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#equalityExpr.
    def visitEqualityExpr(self, ctx:MuParser.EqualityExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#andExpr.
    def visitAndExpr(self, ctx:MuParser.AndExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#parExpr.
    def visitParExpr(self, ctx:MuParser.ParExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#numberAtom.
    def visitNumberAtom(self, ctx:MuParser.NumberAtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#booleanAtom.
    def visitBooleanAtom(self, ctx:MuParser.BooleanAtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#idAtom.
    def visitIdAtom(self, ctx:MuParser.IdAtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#stringAtom.
    def visitStringAtom(self, ctx:MuParser.StringAtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by MuParser#nilAtom.
    def visitNilAtom(self, ctx:MuParser.NilAtomContext):
        return self.visitChildren(ctx)



del MuParser