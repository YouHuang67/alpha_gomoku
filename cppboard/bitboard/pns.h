#ifndef PNS
#define PNS

#include "utils.h"
#include "board.h"
#include "vct_utils.h"
#include "node.h"

const unsigned int PROOF_INF = 10000000;

enum PNSType { OR = 0, AND };

class PNSNode;
class PNSVCTNode;

class PNSNode : public VCTNode<PNSNode> 
{
    public:
        static void InitializeSearch();
        static PNSNode* ProofNumberSearch(const Board& _board, 
                                          StoneType _attacker, 
                                          int maxNodeNum = 1000000);
        inline static int LeafCount() { return leafCount; }
        inline static void ResetLeafCount() { leafCount = 1; }
        inline static int MoveCount() { return moveCount; }

    protected:
        U64 key;
        PNSType type;
        BoardValue value = NONE_VALUE;
        int level = 0;
        int proofNum = 1, disproofNum = 1;
        int mostProvingIndex = -1;
        static VCTBoard* boardPtr;
        static StoneType attacker;
        static U64HashTable<PNSNode>* nodeTablePtr;
        static PNSNode* seenNodes[STONE_NUM];
        static U32 seenActPlys[STONE_NUM];
        static int forwardLevelNum;
        static int backwardLevelNum;
        static int leafCount;
        static int moveCount;
        static PNSNode* GetRoot();
        static PNSNode* GetNode(U64 k, PNSType t, int l = 0, 
                                int pn = 1, int dn = 1);
        void Evaluate();
        void SetProofAndDisproofNumber();
        PNSNode* Forward();
        static PNSNode* Backward(PNSNode* leafNode);
        void Develop(int pn = 1, int dn = 1);
        friend class PNSVCTNode;

};

class PNSVCTNode
{
    public:
        PNSVCTNode() {};
        PNSVCTNode(PNSNode* node) { SetChildren(node); }
        ~PNSVCTNode();
        PNSVCTNode* Next(UC act);
        UC GetAttackAction() const;

    private:
        UC* actions = nullptr;
        PNSVCTNode* children = nullptr;
        void SetChildren(PNSNode* node);

};

inline
PNSVCTNode* PNSVCT(const Board& board, StoneType attacker, 
                   int maxNodeNum = 1000000)
{
    PNSNode::InitializeSearch();
    PNSNode* rootPtr = PNSNode::ProofNumberSearch(board, attacker, maxNodeNum);
    if (rootPtr) return new PNSVCTNode(rootPtr);
    else return nullptr; 
}

inline
void PNSNode::InitializeSearch()
{
    InitializeArrays();
    ResetArrays();
    if (!nodeTablePtr) nodeTablePtr = new U64HashTable<PNSNode>((1 << 10) - 1);
    nodeTablePtr->Reset();
    leafCount = 0;
}

inline
PNSNode* PNSNode::GetRoot()
{
    U64 key = boardPtr->Key();
    PNSNode** handle = nodeTablePtr->FindHandle(key);
    if (!(*handle))
    {
        *handle = GetNode(key, OR);
        (*handle)->Evaluate();
        (*handle)->SetProofAndDisproofNumber();
    }
    return *handle;
}

inline
PNSNode* PNSNode::ProofNumberSearch(const Board& _board, 
                                    StoneType _attacker, 
                                    int maxNodeNum)
{
    if (!boardPtr) boardPtr = new VCTBoard(_board);
    else *boardPtr = _board;
    attacker = _attacker;
    int startNodeNum = VCTNode<PNSNode>::GetNodeCount();
    PNSNode* rootNodePtr = GetRoot();
    PNSNode* nodePtr = rootNodePtr;
    forwardLevelNum = 0;
    backwardLevelNum = 0;
    while (rootNodePtr->proofNum && rootNodePtr->disproofNum &&
           VCTNode<PNSNode>::GetNodeCount() - startNodeNum <= maxNodeNum)
    {
        nodePtr = nodePtr->Forward();
        nodePtr->Develop();
        nodePtr = Backward(nodePtr);
    }
    if (rootNodePtr->proofNum) return nullptr;
    else return rootNodePtr;
}

inline
PNSNode* PNSNode::Forward()
{
    seenNodes[level] = this;
    if (ChildrenNum())
    {
        UC action = Action(mostProvingIndex);
        seenActPlys[level] = (static_cast<U32>(action) << 1) ^ 
                             (static_cast<U32>(level & 1) ^
                              static_cast<U32>(attacker));
        forwardLevelNum++;
        return Child(mostProvingIndex)->Forward();
    }
    static StoneType plys[STONE_NUM];
    static UC acts[STONE_NUM + 1];
    acts[0] = 0;
    for (int i = level - forwardLevelNum; i < level; i++)
    {
        plys[acts[0]] = static_cast<StoneType>(seenActPlys[i] & 1);
        acts[++acts[0]] = static_cast<UC>(seenActPlys[i] >> 1);
    }
    boardPtr->UndoAndMove(backwardLevelNum, acts, plys);
    forwardLevelNum = 0;
    moveCount++;
    return this;
}

inline
PNSNode* PNSNode::Backward(PNSNode* leafNode)
{
    static PNSNode* nodeQueue[1 << 15];
    int start = 0, end = 0;
    nodeQueue[end++] = leafNode;
    int nextLevelStart = 0;
    while (start < end)
    {
        PNSNode* nodePtr = nodeQueue[start++];
        int proofNum = nodePtr->proofNum;
        int disproofNum = nodePtr->disproofNum;
        nodePtr->SetProofAndDisproofNumber();
        if (!nodePtr->ParentsNum()) continue;
        if (proofNum != nodePtr->proofNum ||
            disproofNum != nodePtr->disproofNum)
        {
            if (nextLevelStart < start) nextLevelStart = end;
            int nextLevelEnd = end;
            int num = nodePtr->ParentsNum();
            for (int i = 0; i < num; i++)
            {
                PNSNode* ptr = nodePtr->Parent(i);
                bool exist = false;
                for (int j = nextLevelStart; j < nextLevelEnd; j++)
                    if (ptr == nodeQueue[j])
                    {
                        exist = true;
                        break;
                    }
                if (!exist) nodeQueue[end++] = ptr;
            }
        }
    }
    int startLevel = leafNode->level - 1;
    int endLevel = nodeQueue[end - 1]->level;
    backwardLevelNum = startLevel - endLevel + 1;
    return seenNodes[endLevel];
}

inline
void PNSNode::Develop(int pn, int dn)
{
    PNSType childType = OR == type ? AND : OR;
    StoneType ply = boardPtr->GetPlayer();
    int actionsNum = ActionsNum();
    VCTBoard currentBoard(*boardPtr);
    for (int i = 0; i < actionsNum; i++)
    {
        UC act = Action(i);
        U64 k = Board::UpdateZobristKey(key, ply, act);
        PNSNode** handle = nodeTablePtr->FindHandle(k);
        if (!(*handle)) 
        {
            *handle = GetNode(k, childType, level + 1, pn, dn);
            boardPtr->Move(act, ply);
            (*handle)->Evaluate();
            (*handle)->SetProofAndDisproofNumber();
            *boardPtr = currentBoard;
            if (UNKNOWN == (*handle)->value) leafCount++;
            moveCount += 1;
        }
        SetChild(*handle);
        (*handle)->SetParent(this);
    }
    leafCount--;
}

inline
void PNSNode::Evaluate()
{
    static UC acts[STONE_NUM + 1];
    value = boardPtr->Evaluate(attacker, acts);
    SetActions(acts);
    if (1 == acts[0]) mostProvingIndex = 0;
}

inline
void PNSNode::SetProofAndDisproofNumber()
{
    if (UNKNOWN == value)
    {
        if (!ChildrenNum()) return;
        mostProvingIndex = 0;
        PNSNode* child = Child(mostProvingIndex);
        proofNum = child->proofNum;
        disproofNum = child->disproofNum;
        int actionsNum = ActionsNum();
        if (OR == type)
        {
            for (int i = 1; i < actionsNum; i++)
            {
                child = Child(i);
                disproofNum += child->disproofNum;
                if (proofNum > child->proofNum)
                {
                    proofNum = child->proofNum;
                    mostProvingIndex = i;
                }
            }
        }
        else
        {
            for (int i = 1; i < actionsNum; i++)
            {
                child = Child(i);
                proofNum += child->proofNum;
                if (disproofNum > child->disproofNum)
                {
                    disproofNum = child->disproofNum;
                    mostProvingIndex = i;
                }
            }
        }
    }
    else if (POSITIVE == value)
    {
        proofNum = 0;
        disproofNum = PROOF_INF;
    }
    else if (NEGATIVE == value)
    {
        disproofNum = 0;
        proofNum = PROOF_INF;
    }
}

inline
PNSNode* PNSNode::GetNode(U64 k, PNSType t, int l, int pn, int dn)
{
    PNSNode* nodePtr = VCTNode<PNSNode>::GetNode();
    nodePtr->key = k;
    nodePtr->type = t;
    nodePtr->value = NONE_VALUE;
    nodePtr->proofNum = pn;
    nodePtr->disproofNum = dn;
    nodePtr->level = l;
    nodePtr->mostProvingIndex = -1;
    return nodePtr;
}

inline
void PNSVCTNode::SetChildren(PNSNode* node)
{
    if (!node->ActionsNum()) return;
    UC* pnsActions = node->Actions();
    if (OR == node->type)
    {
        int index = node->mostProvingIndex;
        UC act = pnsActions[index + 1];
        actions = new UC[2];
        actions[0] = 1;
        actions[1] = act;
        children = new PNSVCTNode[1];
        if (node->ChildrenNum())
            children[0].SetChildren(node->Child(index));
    }
    else
    {
        unsigned int num = pnsActions[0];
        actions = new UC[num + 1];
        memcpy(actions, pnsActions, sizeof(UC) * (num + 1));
        children = new PNSVCTNode[num];
        for (int i = 0; i < num; i++) 
            children[i].SetChildren(node->Child(i));
    }
}

inline
PNSVCTNode* PNSVCTNode::Next(UC act)
{
    if (!actions)
    {
        // cout << "no children" << endl;
        return nullptr;
    }
    int index = 1;
    for (; index <= actions[0]; index++)
        if (actions[index] == act) break;
    if (index > actions[0])
    {
        cout << "not found the action: (";
        cout << static_cast<int>(act >> 4) << " ";
        cout << static_cast<int>(act & 15) << ") in the PNS vct tree";
        return nullptr;
    }
    return children + (index - 1);
}

inline
UC PNSVCTNode::GetAttackAction() const
{
    if (!actions || actions[0] > 1)
    {
        cout << "error: no attack action" << endl;
        exit(0);
    }
    return actions[1];
}

#endif