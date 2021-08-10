#ifndef BOARD_WRAPPER
#define BOARD_WRAPPER

#include <vector>
#include "vct_utils.h"
#include "pns.h"

typedef std::vector<int> IntVector;

class BoardWrapper
{
    public:
        BoardWrapper() {}
        BoardWrapper(const BoardWrapper& boardWrapper) 
            { memcpy(this, &boardWrapper, sizeof(BoardWrapper)); }
        IntVector Evaluate(int maxNodeNum = 100000);
        IntVector GetActions(int ply, int sp, int it) const;
        void Move(int act);
        bool IsOver() const { return board.IsOver(); }
        int Winner() const { return static_cast<int>(board.Winner()); }
        int Player() const { return static_cast<int>(board.GetPlayer()); }
        U64 Key() const { return board.Key(); }

    private:
        VCTBoard board;
        StoneType attacker = EMPTY;
        PNSVCTNode* vctRoot = nullptr;
        PNSVCTNode* vctNode = nullptr;
        static IntVector UCsToInts(UC* UCs);

};

inline
void BoardWrapper::Move(int act)
{
    UC action = static_cast<UC>(act);
    board.Move(action);
    if (vctNode)
    {
        vctNode = vctNode->Next(action);
        if (!vctNode)
        {
            delete vctRoot;
            vctRoot = nullptr;
        } 
    } 
}

inline
IntVector BoardWrapper::Evaluate(int maxNodeNum)
{
    StoneType player = board.GetPlayer();
    static UC actions[STONE_NUM + 1];
    actions[0] = 0;
    if (EMPTY == attacker)
    {
        vctRoot = PNSVCT(board, player, maxNodeNum);
        vctNode = vctRoot;
        if (vctRoot) attacker = player;
    }
    IntVector vec;
    if (player == attacker && vctNode)
    {
        UC action = vctNode->GetAttackAction();
        if (action != 0xff)
        {   
            vec.push_back(static_cast<int>(action));
            return vec;
        }
    }
    if (board.GetActions(player, OPEN_FOUR, true, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, FOUR, true, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, OPEN_FOUR, false, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, FOUR, false, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, OPEN_THREE, true, actions))
        return UCsToInts(actions);
    if (board.GetActions(player, OPEN_THREE, false, actions)) 
    {
        board.GetActions(player, THREE, true, actions);
        FilterReplicatedActions(actions);
        return UCsToInts(actions);
    }
    if (POSITIVE == board.Evaluate(player, actions)) 
        return UCsToInts(actions);
    else return vec;
}

inline
IntVector BoardWrapper::GetActions(int ply, int sp, int it) const
{
    IntVector vec;
    static UC actions[STONE_NUM + 1];
    actions[0] = 0;
    if (board.GetActions(static_cast<StoneType>(ply), 
                         static_cast<ShapeType>(sp), 
                         static_cast<bool>(it), actions))
        vec = UCsToInts(actions);
    return vec;
}

inline
IntVector BoardWrapper::UCsToInts(UC* UCs)
{
    IntVector vec(UCs[0], 0);
    for (int i = 1; i <= UCs[0]; i++) 
        vec[i - 1] = static_cast<int>(UCs[i]);
    return vec;
}

#endif