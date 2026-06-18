#include <iostream>
#include <sstream>
#include <cstdint>
#include <cstdlib>

#include "./state.hpp"
#include "config.hpp"
#include "../../policy/game_history.hpp"


/*============================================================
 * KP (King-Piece) Evaluation tables
 *============================================================*/

static const int kp_material[7] = {0, 20, 60, 70, 80, 200, 1000}; //基本棋子價值
static const int simple_material[7] = {0, 2, 6, 7, 8, 20, 100};

// Piece-Square Tables (white/player-0 perspective: row 0 = black backrank, row 5 = white backrank)
// For player 0 (white): pieces start at rows 4-5, advance toward row 0
// For player 1 (black): mirror rows
static const int pst[6][BOARD_H][BOARD_W] = {
    // Pawn: reward advancement
    {{ 0,  0,  0,  0,  0},   // row 0 (promotion row for white, never reached as pawn)
     {15, 15, 15, 15, 15},   // row 1 (one step from promotion)
     { 4,  6, 10,  6,  4},   // row 2
     { 2,  4,  6,  4,  2},   // row 3
     { 0,  2,  2,  2,  0},   // row 4 (starting row for white pawns)
     { 0,  0,  0,  0,  0}},  // row 5 (white backrank)
    // Rook
    {{ 2,  2,  2,  2,  2},
     { 4,  4,  4,  4,  4},
     { 0,  0,  2,  0,  0},
     { 0,  0,  2,  0,  0},
     { 0,  0,  2,  0,  0},
     { 0,  0,  0,  0,  0}},
    // Knight: prefer center
    {{-4, -2,  0, -2, -4},
     {-2,  2,  4,  2, -2},
     { 0,  4,  6,  4,  0},
     { 0,  4,  6,  4,  0},
     {-2,  2,  4,  2, -2},
     {-4, -2,  0, -2, -4}},
    // Bishop: prefer center diagonals
    {{-2,  0,  0,  0, -2},
     { 0,  3,  4,  3,  0},
     { 0,  4,  4,  4,  0},
     { 0,  4,  4,  4,  0},
     { 0,  3,  4,  3,  0},
     {-2,  0,  0,  0, -2}},
    // Queen: prefer center
    {{-2,  0,  2,  0, -2},
     { 0,  2,  4,  2,  0},
     { 0,  4,  6,  4,  0},
     { 0,  4,  6,  4,  0},
     { 0,  2,  4,  2,  0},
     {-2,  0,  2,  0, -2}},
    // King: hide in endgame, stay on backrank early
    {{-8, -8, -8, -8, -8},
     {-4, -4, -4, -4, -4},
     {-4, -4, -4, -4, -4},
     {-4, -4, -4, -4, -4},
     { 4,  4,  0,  4,  4},
     { 6,  6,  2,  6,  6}},
};

static const int tropism_w[7] = {0, 0, 3, 3, 2, 5, 0};

static int king_tropism(
    int piece_type,
    int pr, int pc,
    int ekr, int ekc
){
    int dist = std::max(std::abs(pr - ekr), std::abs(pc - ekc));
    if(dist <= 2){
        return tropism_w[piece_type] * (3 - dist);
    }
    return 0;
}


/*============================================================
 * evaluate()
 *============================================================*/

int State::evaluate(
    bool use_kp_eval, //進階評估
    bool use_mobility, //行動力
    const GameHistory* history //棋局歷史，這裡傳進來但沒用到，只是為了符合介面
){
    (void)history;

    // [TODO 1-1] Win/loss terminal
    if(this->game_state == WIN){
        return P_MAX; //P_MAX 在 base_state.hpp 定義，值是 100000。
    }

    auto self_board  = this->board.board[this->player];
    auto oppn_board  = this->board.board[1 - this->player];
    int self_score = 0, oppn_score = 0;
    int self_kr = -1, self_kc = -1;
    int oppn_kr = -1, oppn_kc = -1;
    /*
    為甚麼先use_kp_eva: 除了king距離，還算升變威脅
    接著看simple material
    然後才use_mobility: 除了mobility差距，還算capture威脅
    升變威脅，和capture計算的時機是甚麼? 為什麼先 use_kp_eva、simple material、use_mobility
    Capture threat，需要掃描 legal_actions。
    */
    if(use_kp_eval){ //use_kp_eva甚麼時候是true?  minimax.hpp 永遠是 true
        // [TODO 1-3] Find king positions
        
        // 找到king == 6
        for(int r = 0; r < BOARD_H; r++){
            for(int c = 0; c < BOARD_W; c++){
                if(self_board[r][c] == 6){ self_kr = r; self_kc = c; }
                if(oppn_board[r][c] == 6){ oppn_kr = r; oppn_kc = c; }
            }
        }

        // [TODO 1-4] Material + PST + tropism
        for(int r = 0; r < BOARD_H; r++){
            for(int c = 0; c < BOARD_W; c++){
                int sp = self_board[r][c]; // 0-6
                if(sp > 0){
                    // PST: for self (player), row increases toward enemy
                    // player=0 (white): rows go 0..5, advances toward row 0 -> already correct
                    // player=1 (black): mirror row
                    int pst_row = (this->player == 0) ? r : (BOARD_H - 1 - r);
                    self_score += kp_material[sp] + pst[sp-1][pst_row][c];
                    if(oppn_kr >= 0 && sp != 6){ //為甚麼看opp king的r?
                        self_score += king_tropism(sp, r, c, oppn_kr, oppn_kc);
                    }
                }

                int op = oppn_board[r][c];
                if(op > 0){
                    int opp_pst_row = (this->player == 1) ? r : (BOARD_H - 1 - r);
                    oppn_score += kp_material[op] + pst[op-1][opp_pst_row][c];
                    if(self_kr >= 0 && op != 6){ //oppn_kr >= 0：確認敵方 King 還在棋盤上（沒有被吃掉）sp != 6：我方 King 不計算 tropism（King 不該冒險去靠近敵王）
                        oppn_score += king_tropism(op, r, c, self_kr, self_kc);
                    }
                }
            }
        }
        //升變威脅
        // Pawn promotion threat: bonus for pawns close to promotion
        for(int r = 0; r < BOARD_H; r++){
            for(int c = 0; c < BOARD_W; c++){
                if(self_board[r][c] == 1){//兵
                    // player=0: promotes at row 0; distance = r
                    // player=1: promotes at row BOARD_H-1; distance = BOARD_H-1-r
                    int dist = (this->player == 0) ? r : (BOARD_H - 1 - r);
                    if(dist == 1) self_score += 80; //在幾步就可升變
                    else if(dist == 2) self_score += 40;
                    else if(dist == 3) self_score += 20;
                }
                if(oppn_board[r][c] == 1){
                    int dist = (this->player == 1) ? r : (BOARD_H - 1 - r);
                    if(dist == 1) oppn_score += 80;
                    else if(dist == 2) oppn_score += 40;
                    else if(dist == 3) oppn_score += 20;
                }
            }
        }

    } else {
        // [TODO 1-2] Simple material //simple_material 在幹嘛? 跟kp_material為什麼不一樣?
        for(int r = 0; r < BOARD_H; r++){
            for(int c = 0; c < BOARD_W; c++){
                int sp = self_board[r][c];
                if(sp > 0) self_score += simple_material[sp];
                int op = oppn_board[r][c];
                if(op > 0) oppn_score += simple_material[op];
            }
        }
    }

    int bonus = 0;

    // [TODO 1-5] Mobility bonus
    if(use_mobility){ //use_mobility甚麼時候是true?
        // Current state already has legal_actions
        int self_mobility = (int)this->legal_actions.size();

        // Compute opponent mobility via null state
        State opp_state(this->board, 1 - this->player); //現在是我的回合，this->legal_actions 已經有我的合法步了，所以要「假裝換對手下」
        opp_state.get_legal_actions();
        int oppn_mobility = (int)opp_state.legal_actions.size();

        bonus += 2 * (self_mobility - oppn_mobility);

        // Capture threat bonus: if any of our moves captures a high-value piece
        for(auto& action : this->legal_actions){
            int tr = (int)action.second.first; //second 和 first是神麼意思?
            int tc = (int)action.second.second;
            int cap = oppn_board[tr][tc]; //我這步走完之後，目標格上有沒有對手的棋子」，有的話就加分。
            if(cap > 0){
                bonus += simple_material[cap]/4;
            }
        }
    }
    if(this->step >= 70){
        // 用比賽規則的子數計算
        static const int endgame_val[7] = {0, 2, 6, 7, 8, 20, 0};
        int self_end = 0, oppn_end = 0;
        for(int r = 0; r < BOARD_H; r++){
            for(int c = 0; c < BOARD_W; c++){
                int sp = self_board[r][c];
                int op = oppn_board[r][c];
                if(sp > 0) self_end += endgame_val[sp];
                if(op > 0) oppn_end += endgame_val[op];
            }
        }
        // 殘局加重比子數的權重
        int endgame_weight = (this->step - 70) * 2;
        bonus += (self_end - oppn_end) * endgame_weight;
    }
    // 殘局時 King 應該主動進攻
    if(this->step >= 75){
        // 鼓勵我方 King 靠近敵方 King
        if(self_kr >= 0 && oppn_kr >= 0){
            int king_dist = std::abs(self_kr - oppn_kr) + 
                            std::abs(self_kc - oppn_kc);
            bonus += (7 - king_dist) * 10;
        }
    }
    return self_score - oppn_score + bonus;
}


/*============================================================
 * Zobrist hash
 *============================================================*/
static uint64_t zobrist_piece[2][7][BOARD_H][BOARD_W];
static uint64_t zobrist_side;
static bool zobrist_ready = false;

static void init_zobrist(){
    uint64_t s = 0x7A35C9D1E4F02B68ULL;
    auto rand64 = [&s]() -> uint64_t {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
    };
    for(int p = 0; p < 2; p++){
        for(int t = 0; t < 7; t++){
            for(int r = 0; r < BOARD_H; r++){
                for(int c = 0; c < BOARD_W; c++){
                    zobrist_piece[p][t][r][c] = rand64();
                }
            }
        }
    }
    zobrist_side = rand64();
    zobrist_ready = true;
}

uint64_t State::compute_hash_full() const{
    if(!zobrist_ready){ init_zobrist(); }
    uint64_t h = 0;
    for(int p = 0; p < 2; p++){
        for(int r = 0; r < BOARD_H; r++){
            for(int c = 0; c < BOARD_W; c++){
                int piece = this->board.board[p][r][c];
                if(piece){
                    h ^= zobrist_piece[p][piece][r][c];
                }
            }
        }
    }
    if(this->player){ h ^= zobrist_side; }
    return h;
}


State* State::next_state(const Move& move){
    if(!zobrist_ready){ init_zobrist(); }

    Board next = this->board;
    Point from = move.first, to = move.second;
    int p = this->player;
    int opp = 1 - p;

    int8_t orig_piece = next.board[p][from.first][from.second];
    int8_t moved = orig_piece;
    if(moved == 1 && (to.first == BOARD_H-1 || to.first == 0)){
        moved = 5; // pawn promotion to queen
    }

    uint64_t h = this->hash();
    h ^= zobrist_side;
    h ^= zobrist_piece[p][orig_piece][from.first][from.second];

    int8_t captured = next.board[opp][to.first][to.second];
    if(captured){
        h ^= zobrist_piece[opp][captured][to.first][to.second];
        next.board[opp][to.first][to.second] = 0;
    }

    h ^= zobrist_piece[p][moved][to.first][to.second];
    next.board[p][from.first][from.second] = 0;
    next.board[p][to.first][to.second] = moved;

    State* ns = new State(next, opp);
    ns->zobrist_hash = h;
    ns->zobrist_valid = true;
    return ns;
}


static const int move_table_rook_bishop[8][7][2] = {
  {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}},
  {{0, -1}, {0, -2}, {0, -3}, {0, -4}, {0, -5}, {0, -6}, {0, -7}},
  {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}},
  {{-1, 0}, {-2, 0}, {-3, 0}, {-4, 0}, {-5, 0}, {-6, 0}, {-7, 0}},
  {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}},
  {{1, -1}, {2, -2}, {3, -3}, {4, -4}, {5, -5}, {6, -6}, {7, -7}},
  {{-1, 1}, {-2, 2}, {-3, 3}, {-4, 4}, {-5, 5}, {-6, 6}, {-7, 7}},
  {{-1, -1}, {-2, -2}, {-3, -3}, {-4, -4}, {-5, -5}, {-6, -6}, {-7, -7}},
};

// [TODO 2-1] Knight move table
static const int move_table_knight[8][2] = {
    {-2, -1}, {-2, 1},
    {-1, -2}, {-1, 2},
    { 1, -2}, { 1, 2},
    { 2, -1}, { 2, 1},
};

static const int move_table_king[8][2] = {
  {1, 0}, {0, 1}, {-1, 0}, {0, -1},
  {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
};


/*============================================================
 * Naive move generation
 *============================================================*/
void State::get_legal_actions_naive(){
    this->game_state = NONE;
    std::vector<Move> all_actions;
    all_actions.reserve(64);
    auto self_board = this->board.board[this->player];
    auto oppn_board = this->board.board[1 - this->player];

    int now_piece, oppn_piece;
    for(int i=0; i<BOARD_H; i++){
        for(int j=0; j<BOARD_W; j++){
            if((now_piece=self_board[i][j])){
                switch(now_piece){
                    case 1: //pawn
                        if(this->player && i<BOARD_H-1){
                            //black advances downward (increasing row)
                            if(!oppn_board[i+1][j] && !self_board[i+1][j]){ // 前面沒有任何棋子 → 可以前進
                                all_actions.push_back(Move(Point(i,j), Point(i+1,j)));
                            }
                            if(j<BOARD_W-1 && (oppn_piece=oppn_board[i+1][j+1])>0){// 右前方有敵方棋子 → 可以斜吃
                                all_actions.push_back(Move(Point(i,j), Point(i+1,j+1)));
                                if(oppn_piece==6){ this->game_state=WIN; this->legal_actions=all_actions; return; }// 吃到敵方King → 直接WIN，立刻回傳
                            }
                            if(j>0 && (oppn_piece=oppn_board[i+1][j-1])>0){// 左前方有敵方棋子 → 可以斜吃
                                all_actions.push_back(Move(Point(i,j), Point(i+1,j-1)));
                                if(oppn_piece==6){ this->game_state=WIN; this->legal_actions=all_actions; return; }
                            }
                        } else if(!this->player && i>0){
                            //white advances upward (decreasing row)// 白方兵：往上走（row 減少），邏輯鏡像
                            if(!oppn_board[i-1][j] && !self_board[i-1][j]){
                                all_actions.push_back(Move(Point(i,j), Point(i-1,j)));
                            }
                            if(j<BOARD_W-1 && (oppn_piece=oppn_board[i-1][j+1])>0){
                                all_actions.push_back(Move(Point(i,j), Point(i-1,j+1)));
                                if(oppn_piece==6){ this->game_state=WIN; this->legal_actions=all_actions; return; }
                            }
                            if(j>0 && (oppn_piece=oppn_board[i-1][j-1])>0){
                                all_actions.push_back(Move(Point(i,j), Point(i-1,j-1)));
                                if(oppn_piece==6){ this->game_state=WIN; this->legal_actions=all_actions; return; }
                            }
                        }
                        break;

                    case 2: //rook
                    case 4: //bishop
                    case 5: { //queen
                        int st, end;
                        switch(now_piece){
                            case 2: st=0; end=4; break; // 方向 0~3（直線）
                            case 4: st=4; end=8; break; // 方向 4~7（斜線）
                            default: st=0; end=8; break;// 方向 0~7（全部）  
                        }
                        for(int part=st; part<end; part++){
                            auto move_list = move_table_rook_bishop[part];
                            for(int k=0; k<std::max(BOARD_H,BOARD_W); k++){
                                int p[2] = {move_list[k][0]+i, move_list[k][1]+j}; 
                                if(p[0]>=BOARD_H||p[0]<0||p[1]>=BOARD_W||p[1]<0) break;//越界，break 是因為是滑動棋子，需要一格一格檢查中間有沒有阻擋
                                if(self_board[p[0]][p[1]]) break;// 遇到自己的棋子 → 停止
                                all_actions.push_back(Move(Point(i,j), Point(p[0],p[1])));
                                oppn_piece = oppn_board[p[0]][p[1]];
                                if(oppn_piece){// 遇到敵方棋子 → 可以吃，但吃完停止
                                    if(oppn_piece==6){ this->game_state=WIN; this->legal_actions=all_actions; return; }
                                    else break;
                                }
                            }
                        }
                        break;
                    }

                    case 3: //knight [TODO 2-2]
                        for(auto& mv : move_table_knight){
                            int p[2] = {mv[0]+i, mv[1]+j};
                            if(p[0]>=BOARD_H||p[0]<0||p[1]>=BOARD_W||p[1]<0) continue; //越界，不是滑動棋子，不需要一格一格檢查中間有沒有阻擋
                            if(self_board[p[0]][p[1]]) continue; //自己的棋子跳過
                            all_actions.push_back(Move(Point(i,j), Point(p[0],p[1])));
                            oppn_piece = oppn_board[p[0]][p[1]];
                            if(oppn_piece==6){ this->game_state=WIN; this->legal_actions=all_actions; return; }
                        }
                        // fall through intentionally removed; knight has its own break
                        break;

                    case 6: //king
                        for(auto& mv : move_table_king){
                            int p[2] = {mv[0]+i, mv[1]+j};
                            if(p[0]>=BOARD_H||p[0]<0||p[1]>=BOARD_W||p[1]<0) continue;
                            if(self_board[p[0]][p[1]]) continue;
                            all_actions.push_back(Move(Point(i,j), Point(p[0],p[1])));
                            oppn_piece = oppn_board[p[0]][p[1]];
                            if(oppn_piece==6){ this->game_state=WIN; this->legal_actions=all_actions; return; } //not possible

                        }
                        break;
                }
            }
        }
    }
    this->legal_actions = all_actions;
}


/*============================================================
 * Bitboard move generation
 *============================================================*/

#define BB_SQ(r,c)  ((r)*BOARD_W+(c))
#define BB_ROW(sq)  ((sq)/BOARD_W)
#define BB_COL(sq)  ((sq)%BOARD_W)

static const int bb_dr[8] = {0, 0, 1, -1, 1, 1, -1, -1};
static const int bb_dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};

static bool bb_tables_ready = false;
static uint32_t bb_pawn_push[2][BOARD_H*BOARD_W];
static uint32_t bb_pawn_cap[2][BOARD_H*BOARD_W];
static uint32_t bb_knight[BOARD_H*BOARD_W];
static uint32_t bb_king[BOARD_H*BOARD_W];

static void init_bb_tables(){
    for(int r=0; r<BOARD_H; r++){
        for(int c=0; c<BOARD_W; c++){
            int sq = BB_SQ(r,c);
            bb_pawn_push[0][sq] = 0;
            bb_pawn_push[1][sq] = 0;
            bb_pawn_cap[0][sq] = 0; //斜吃
            bb_pawn_cap[1][sq] = 0;
            bb_knight[sq] = 0;
            bb_king[sq] = 0;

            // white pawn (player 0): advances to r-1
            if(r > 0){ bb_pawn_push[0][sq] |= 1u << BB_SQ(r-1,c); }
            if(r > 0 && c > 0)    bb_pawn_cap[0][sq] |= 1u << BB_SQ(r-1,c-1);
            if(r > 0 && c < BOARD_W-1) bb_pawn_cap[0][sq] |= 1u << BB_SQ(r-1,c+1);

            // black pawn (player 1): advances to r+1
            if(r < BOARD_H-1){ bb_pawn_push[1][sq] |= 1u << BB_SQ(r+1,c); }
            if(r < BOARD_H-1 && c > 0)    bb_pawn_cap[1][sq] |= 1u << BB_SQ(r+1,c-1);
            if(r < BOARD_H-1 && c < BOARD_W-1) bb_pawn_cap[1][sq] |= 1u << BB_SQ(r+1,c+1);

            // knight
            static const int knd[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
            for(auto& km : knd){
                int nr = r+km[0], nc = c+km[1];
                if(nr>=0&&nr<BOARD_H&&nc>=0&&nc<BOARD_W) //in board
                    bb_knight[sq] |= 1u << BB_SQ(nr,nc);
            }

            // king
            for(int d=0; d<8; d++){
                int nr=r+bb_dr[d], nc=c+bb_dc[d];
                if(nr>=0&&nr<BOARD_H&&nc>=0&&nc<BOARD_W)
                    bb_king[sq] |= 1u << BB_SQ(nr,nc);
            }
        }
    }
    bb_tables_ready = true;
}

void State::get_legal_actions_bitboard(){
    if(!bb_tables_ready) init_bb_tables();

    this->game_state = NONE;
    this->legal_actions.clear();
    this->legal_actions.reserve(64);

    int self = this->player;
    int opp  = 1 - self;

    // Build piece-type arrays and occupancy bitmasks
    uint8_t self_pt[BOARD_H*BOARD_W] = {};// 我方每個格子的棋子種類
    uint8_t oppn_pt[BOARD_H*BOARD_W] = {};// 對手每個格子的棋子種類
    uint32_t self_occ = 0, oppn_occ = 0, all_occ = 0;

    for(int r=0; r<BOARD_H; r++){
        for(int c=0; c<BOARD_W; c++){
            int sq = BB_SQ(r,c);
            int sp = this->board.board[self][r][c];
            int op = this->board.board[opp][r][c];
            if(sp){ self_pt[sq] = sp; self_occ |= 1u<<sq; }
            if(op){ oppn_pt[sq] = op; oppn_occ |= 1u<<sq; }
        }
    }
    all_occ = self_occ | oppn_occ;

    for(int r=0; r<BOARD_H; r++){
        for(int c=0; c<BOARD_W; c++){
            int sq = BB_SQ(r,c);
            int piece = self_pt[sq];
            if(!piece) continue;

            uint32_t targets = 0;

            switch(piece){
                case 1: { // Pawn
                    uint32_t push = bb_pawn_push[self][sq] & ~all_occ; // & ~all_occ  = 只保留空格
                    uint32_t cap  = bb_pawn_cap[self][sq]  & oppn_occ;
                    uint32_t cap_scan = cap;
                    while(cap_scan){
                        int to = __builtin_ctz(cap_scan); //找到最低位的 1 在第幾個 bit
                        cap_scan &= cap_scan-1;
                        if(oppn_pt[to]==6){
                            this->game_state = WIN;
                            this->legal_actions.push_back(Move(Point(r,c), Point(BB_ROW(to),BB_COL(to))));
                            return;
                        }
                    }
                    targets = push | cap;
                    break;
                }
                case 3: { // Knight
                    targets = bb_knight[sq] & ~self_occ;
                    uint32_t opp_t = targets & oppn_occ;
                    while(opp_t){
                        int to = __builtin_ctz(opp_t); opp_t &= opp_t-1;
                        if(oppn_pt[to]==6){
                            this->game_state = WIN;
                            this->legal_actions.push_back(Move(Point(r,c), Point(BB_ROW(to),BB_COL(to))));
                            return;
                        }
                    }
                    break;
                }
                case 6: { // King
                    targets = bb_king[sq] & ~self_occ;
                    uint32_t opp_t = targets & oppn_occ;
                    while(opp_t){
                        int to = __builtin_ctz(opp_t); opp_t &= opp_t-1;
                        if(oppn_pt[to]==6){
                            this->game_state = WIN;
                            this->legal_actions.push_back(Move(Point(r,c), Point(BB_ROW(to),BB_COL(to))));
                            return;
                        }
                    }
                    break;
                }
                case 2: // Rook
                case 4: // Bishop
                case 5: { // Queen
                    int d_start = (piece==4) ? 4 : 0;
                    int d_end   = (piece==2) ? 4 : 8;
                    for(int d=d_start; d<d_end; d++){
                        int cr=r+bb_dr[d], cc=c+bb_dc[d];
                        while(cr>=0&&cr<BOARD_H&&cc>=0&&cc<BOARD_W){
                            int to = BB_SQ(cr,cc);
                            uint32_t to_bit = 1u<<to;
                            if(self_occ & to_bit) break;
                            if((oppn_occ & to_bit) && oppn_pt[to]==6){
                                this->game_state = WIN;
                                this->legal_actions.push_back(Move(Point(r,c), Point(cr,cc)));
                                return;
                            }
                            targets |= to_bit;
                            if(oppn_occ & to_bit) break;
                            cr+=bb_dr[d]; cc+=bb_dc[d];
                        }
                    }
                    break;
                }
            }

            while(targets){
                int to = __builtin_ctz(targets);
                targets &= targets-1;
                this->legal_actions.push_back(Move(Point(r,c), Point(BB_ROW(to),BB_COL(to))));
            }
        }
    }
}


/*============================================================
 * Dispatcher
 *============================================================*/
void State::get_legal_actions(){
#ifdef USE_BITBOARD
    get_legal_actions_bitboard();
#else
    get_legal_actions_naive();
#endif
}


const char piece_table[2][7][5] = {
  {" ", "♙", "♖", "♘", "♗", "♕", "♔"},
  {" ", "♟", "♜", "♞", "♝", "♛", "♚"}
};

std::string State::encode_output() const{
    std::stringstream ss;
    int now_piece;
    for(int i=0; i<BOARD_H; i++){
        for(int j=0; j<BOARD_W; j++){
            if((now_piece = this->board.board[0][i][j])){
                ss << std::string(piece_table[0][now_piece]);
            } else if((now_piece = this->board.board[1][i][j])){
                ss << std::string(piece_table[1][now_piece]);
            } else {
                ss << " ";
            }
            ss << " ";
        }
        ss << "\n";
    }
    return ss.str();
}

std::string State::encode_state(){
    std::stringstream ss;
    ss << this->player << "\n";
    for(int pl=0; pl<2; pl++){
        for(int i=0; i<BOARD_H; i++){
            for(int j=0; j<BOARD_W; j++){
                ss << int(this->board.board[pl][i][j]) << " ";
            }
            ss << "\n";
        }
        ss << "\n";
    }
    return ss.str();
}

BaseState* State::create_null_state() const{
    State* s = new State(this->board, 1 - this->player);
    s->get_legal_actions();
    return s;
}

static const char* piece_chars = ".PRNBQK";
static const char* piece_chars_lower = ".prnbqk";

std::string State::encode_board() const{
    std::string s;
    for(int r=0; r<BOARD_H; r++){
        if(r>0) s += '/';
        for(int c=0; c<BOARD_W; c++){
            int w = board.board[0][r][c];
            int b = board.board[1][r][c];
            if(w>0&&w<=6)      s += piece_chars[w];
            else if(b>0&&b<=6) s += piece_chars_lower[b];
            else               s += '.';
        }
    }
    return s;
}

void State::decode_board(const std::string& s, int side_to_move){
    player = side_to_move;
    game_state = UNKNOWN;
    zobrist_valid = false;
    board = Board{};
    int r=0, c=0;
    for(char ch : s){
        if(ch=='/'){ r++; c=0; continue; }
        if(r>=BOARD_H||c>=BOARD_W) break;
        if(ch>='A'&&ch<='Z'){
            for(int p=1;p<=6;p++){
                if(piece_chars[p]==ch){ board.board[0][r][c]=p; break; }
            }
        } else if(ch>='a'&&ch<='z'){
            for(int p=1;p<=6;p++){
                if(piece_chars_lower[p]==ch){ board.board[1][r][c]=p; break; }
            }
        }
        c++;
    }
    get_legal_actions();
}

std::string State::cell_display(int row, int col) const{
    int w = static_cast<int>(board.board[0][row][col]);
    int b = static_cast<int>(board.board[1][row][col]);
    if(w){ const char* n=".PRNBQK"; return std::string(" ")+n[w]+" "; }
    if(b){ const char* n=".prnbqk"; return std::string(" ")+n[b]+" "; }
    return " . ";
}
//出現過 3 次以上，判定為重複局
bool State::check_repetition(const GameHistory& history, int& out_score) const {
    if(history.count(hash()) >= 3){ out_score = 0; return true; }
    return false;
}
