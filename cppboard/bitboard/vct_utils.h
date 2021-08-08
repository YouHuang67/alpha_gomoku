#ifndef VCT_UTILS
#define VCT_UTILS

#include <cstdlib>
#include <algorithm>
#include "utils.h"
#include "board.h"

enum BoardValue { UNKNOWN = 0, POSITIVE, NEGATIVE, NONE_VALUE };

class VCTBoard : public Board
{
    public:
        VCTBoard() { Initialize(); }
        VCTBoard(const Board& board) { memcpy(this, &board, sizeof(Board)); }
        VCTBoard(const VCTBoard& board) { memcpy(this, &board, sizeof(Board)); }
        BoardValue Evaluate(StoneType attacker, UC* actions) const;

};

inline
bool FilterReplicatedActions(UC* actions)
{
    if (!actions[0]) return false;
    int num = actions[0];
    std::sort(actions + 1, actions + 1 + num);
    actions[0] = 1;
    bool replicated = false;
    for (int i = 2; i <= num; i++)
        if (actions[i] == actions[actions[0]]) replicated = true;
        else actions[++actions[0]] = actions[i];
    return replicated;
}

inline
BoardValue VCTBoard::Evaluate(StoneType attacker, UC* actions) const
{
    const StoneType player = bit.GetPlayer();
    actions[0] = 0;
    if (player != attacker) 
    {
        if (GetActions(player, OPEN_FOUR, true) |
            GetActions(player, FOUR, true))                  return NEGATIVE;
        if (GetActions(player, OPEN_FOUR, false))            return POSITIVE;
        if (GetActions(player, FOUR, false, actions))        return UNKNOWN;
        if (GetActions(player, OPEN_THREE, true) ||
            !GetActions(player, OPEN_THREE, false, actions)) return NEGATIVE;
        GetActions(player, THREE, true, actions);            return UNKNOWN;
    }
    else
    {
        if (GetActions(player, OPEN_FOUR, true) |
            GetActions(player, FOUR, true))                 return POSITIVE;
        if (GetActions(player, OPEN_FOUR, false))           return NEGATIVE;
        if (GetActions(player, FOUR, false, actions))       return UNKNOWN;
        if (GetActions(player, OPEN_THREE, true))           return POSITIVE;
        if (GetActions(player, OPEN_THREE, false, actions)) 
        {
            GetActions(player, THREE, true, actions);
            FilterReplicatedActions(actions);
            return UNKNOWN;
        }
    }
    static int three = static_cast<int>(THREE);
    static int openTwo = static_cast<int>(OPEN_TWO);
    static UC** actionTable = LineShape::GetActionTable();
    static UC twoActions[STONE_NUM + 1], threeActions[STONE_NUM + 1];
    int ply = static_cast<int>(player);
    twoActions[0] = 0;
    threeActions[0] = 0;
    for (int row = 0; row < BOARD_SIZE; row++)
    {
        int threeIndex = GetShapeIndex(three, ply, 1, row);
        U32 threeCodedActions[4] = {
            codedActions[threeIndex] & CODING_MASK,   
            codedActions[threeIndex] >> CODING_LENGTH, 
            codedActions[threeIndex+1] & CODING_MASK, 
            codedActions[threeIndex+1] >> CODING_LENGTH
        };
        U32 threeOrActions = threeCodedActions[0] | threeCodedActions[1] |
                             threeCodedActions[2] | threeCodedActions[3];
        if (threeOrActions)
        {
            U32 threeXorActions = threeCodedActions[0] ^ threeCodedActions[1] ^
                                  threeCodedActions[2] ^ threeCodedActions[3];
            U32 doubleThree = threeOrActions ^ threeXorActions;
            if (doubleThree)
            {
                UC* handle = actionTable[doubleThree];
                for (int i = 1; i <= handle[0]; i++)
                    actions[++actions[0]] = ActionFlatten(row, handle[i]);
                return POSITIVE;
            }
            UC* handle = actionTable[threeOrActions];
            for (int i = 1; i <= handle[0]; i++)
                threeActions[++threeActions[0]] = ActionFlatten(row, handle[i]);
        }

        int openTwoIndex = GetShapeIndex(openTwo, ply, 1, row);
        U32 openTwoCodedActions[4] = {
            codedActions[openTwoIndex] & CODING_MASK,   
            codedActions[openTwoIndex] >> CODING_LENGTH, 
            codedActions[openTwoIndex+1] & CODING_MASK, 
            codedActions[openTwoIndex+1] >> CODING_LENGTH
        };
        U32 openTwoOrActions = openTwoCodedActions[0] | openTwoCodedActions[1] |
                               openTwoCodedActions[2] | openTwoCodedActions[3];
        if (openTwoOrActions)
        {
            U32 openTwoXorActions = openTwoCodedActions[0] ^ openTwoCodedActions[1] ^
                                    openTwoCodedActions[2] ^ openTwoCodedActions[3];
            U32 doubleOpenTwo = openTwoOrActions ^ openTwoXorActions;
            if (doubleOpenTwo && 
                !GetActions(BLACK == attacker ? WHITE: BLACK, THREE, true))
            {
                UC* handle = actionTable[doubleOpenTwo];
                for (int i = 1; i <= handle[0]; i++)
                    actions[++actions[0]] = ActionFlatten(row, handle[i]);
                return POSITIVE;
            }
            UC* handle = actionTable[openTwoOrActions];
            for (int i = 1; i <= handle[0]; i++)
                twoActions[++twoActions[0]] = ActionFlatten(row, handle[i]);
        }

        U32 threeOrTwo = threeOrActions | openTwoOrActions;
        if (!threeOrTwo) continue;
        U32 threeXorTwo = threeOrActions ^ openTwoOrActions;
        U32 threeTwo = threeOrTwo ^ threeXorTwo;
        if (!threeTwo) continue;
        UC* handle = actionTable[threeTwo];
        for (int i = 1; i <= handle[0]; i++)
            actions[++actions[0]] = ActionFlatten(row, handle[i]);
        return POSITIVE;
    }
    for (int i = 1; i <= twoActions[0]; i++) 
        actions[++actions[0]] = twoActions[i];
    for (int i = 1; i <= threeActions[0]; i++) 
        actions[++actions[0]] = threeActions[i];
    if (actions[0]) return UNKNOWN;
    else return NEGATIVE;
}

#endif