#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <utility>
#include <cstring>

#include "state.hpp"
#include "minimax.hpp"


/* ============================================================
 * Transposition Table
 * ============================================================ */

enum TTFlag : uint8_t {
    TT_EXACT = 0,
    TT_LOWER = 1,   // alpha (fail-low)  – lower bound
    TT_UPPER = 2,   // beta  (fail-high) – upper bound
};

struct TTEntry {
    uint64_t hash   = 0; // 這個盤面的 Zobrist hash（識別用
    int      score  = 0; // 這個盤面算出來的分數
    Move     best_move = {Point(0,0), Point(0,0)}; // 這個盤面最好的步
    int8_t   depth  = 0; // 當時搜尋的深度
    TTFlag   flag   = TT_EXACT; // 分數的類型 exact, alpha(low),beta(high)
    bool     valid  = false;  // 這格有沒有存過東西?
};

static const size_t TT_SIZE = 1 << 22;  // ~4M entries (~200 MB) 開一個很大的陣列存所有記錄
static TTEntry tt_table[TT_SIZE];

static inline size_t tt_index(uint64_t hash){ return hash & (TT_SIZE - 1); } //& (TT_SIZE - 1) 是把 hash 限制在 0 ~ 4百萬之間，概念就像取餘數。

static void tt_clear(){
    //std::memset(tt_table, 0, sizeof(tt_table));
    for(size_t i = 0; i < TT_SIZE; i++){
        tt_table[i] = TTEntry{};
    }
}
//查詢
static const TTEntry* tt_probe(uint64_t hash){
    const TTEntry& e = tt_table[tt_index(hash)];
    if(e.valid && e.hash == hash) return &e;
    return nullptr;
}

static void tt_store(uint64_t hash, int score, int depth, TTFlag flag, const Move& best){
    TTEntry& e = tt_table[tt_index(hash)];// 用hash找到位置
    // Always-replace (simplest, good enough)
    e.hash      = hash;
    e.score     = score;
    e.depth     = (int8_t)depth;
    e.flag      = flag;
    e.best_move = best;
    e.valid     = true;
}


/* ============================================================
 * Move Ordering
 * Scores used for std::sort (higher = earlier)
 * ============================================================ */

// MVV-LVA piece values (same scale as config.hpp PIECE_VALUES)
static const int mvv_val[7] = {0, 10, 50, 30, 30, 90, 900};
/*
1. TT move（上一次搜尋找到的最佳步）
2. 吃子（MVV-LVA：用小棋子吃大棋子最優先）
3. 升變
4. 普通移動
*/
static int move_score(const Move& mv, const State* state, const Move& tt_move){
    // TT move first
    if(mv == tt_move) return 1000000; //上一步

    int fr = (int)mv.first.first,  fc = (int)mv.first.second;// 起點
    int tr = (int)mv.second.first, tc = (int)mv.second.second; // 終點
    int opp = 1 - state->player;

    int captured = state->board.board[opp][tr][tc]; // 被吃的棋子
    int mover    = state->board.board[state->player][fr][fc]; // 我方移動的棋子

    if(captured){
        // MVV-LVA: big victim, small attacker
        return 100000 + mvv_val[captured] * 10 - mvv_val[mover];// 吃高價值棋子 用低價值棋子去吃 = 最優先
    }

    // Promotion
    if(mover == 1){
        bool promotes = (state->player == 0 && tr == 0) ||
                        (state->player == 1 && tr == BOARD_H-1);
        if(promotes) return 50000;
    }

    return 0;// 普通move
}

static void order_moves(std::vector<Move>& moves, const State* state, const Move& tt_move){
    std::sort(moves.begin(), moves.end(),
        [&](const Move& a, const Move& b){
            return move_score(a, state, tt_move) > move_score(b, state, tt_move);
        });
}


/* ============================================================
 * Quiescence Search
 * Only expands captures and promotions to avoid horizon effect
 * ============================================================ */
// 因為 quiescence 不限深度，搜到盤面穩定為止。
static int quiescence(
    State* state,
    int alpha,
    int beta,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ctx.stop) return 0;

    // Stand-pat站立評估
    int stand_pat = state->evaluate(p.use_kp_eval, false, nullptr);//什麼都不做，當前盤面的分數
    if(stand_pat >= beta) return beta; //剪
    if(stand_pat > alpha) alpha = stand_pat; //什麼都不做」比之前找到的步還好

    // Only look at captures / promotions
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }
    if(state->game_state == WIN) return P_MAX - ply; //目前搜尋到第幾層
    if(state->game_state == DRAW) return 0;

    int opp = 1 - state->player;

    // Collect noisy moves (captures + promotions) noisy moves（吃子＋升變）
    std::vector<Move> noisy;
    noisy.reserve(16);
    for(auto& mv : state->legal_actions){
        int tr = (int)mv.second.first, tc = (int)mv.second.second;
        int captured = state->board.board[opp][tr][tc];
        int mover    = state->board.board[state->player][(int)mv.first.first][(int)mv.first.second];
        bool is_promo = (mover==1) && ((state->player==0 && tr==0)||(state->player==1 && tr==BOARD_H-1));
        if(captured || is_promo) noisy.push_back(mv);
    }

    //Order by MVV-LVA + Promotion
    static const Move null_move = {Point(0,0),Point(0,0)};
    order_moves(noisy, state, null_move);

    for(auto& mv : noisy){
        if(ctx.stop) break;
        State* child = state->next_state(mv);
        if(child->legal_actions.empty() && child->game_state == UNKNOWN){
            child->get_legal_actions();
        }

        int raw;
        if(child->same_player_as_parent()){
            raw = quiescence(child, alpha, beta, ply+1, ctx, p);
        } else {
            raw = -quiescence(child, -beta, -alpha, ply+1, ctx, p);
        }
        delete child;

        if(raw >= beta){ return beta; }
        if(raw > alpha){ alpha = raw; }
    }
    return alpha;
}


/* ============================================================
 * PVS / Alpha-Beta core
 * ============================================================ */

static int pvs(
    State*       state,
    int          depth,
    int          alpha,
    int          beta,
    GameHistory& history,
    int          ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply; // 記錄最深到幾層
    if(ctx.stop) return 0; // 時間到，立刻停止，ctx.stop在哪裡可以判斷時間到

    /* === Lazy move generation === */ //先查TT → 命中就直接回傳，不用算合法步
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;   // prefer faster wins
    }
    if(state->game_state == DRAW){
        return 0;
    }
   
    /* === Repetition === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    uint64_t h = state->hash();// 不是走過的路徑，棋子的位置

    /* === Transposition table probe === */
    Move tt_best = {Point(0,0), Point(0,0)};
    const TTEntry* tte = tt_probe(h); //查相同局面
    if(tte && tte->depth >= depth){
        int s = tte->score;//depth 夠深 → 直接用分數
        if(tte->flag == TT_EXACT)                   return s; //直接用已知的精確答案，不需要重新計算。
        if(tte->flag == TT_LOWER && s >= beta)      return s; 
        if(tte->flag == TT_UPPER && s <= alpha)     return s;
        tt_best = tte->best_move;
    } else if(tte){ //depth 不夠深，就算深度不同，「這個盤面最好的步」通常還是一樣的
        tt_best = tte->best_move;
    }

    /* === Leaf: quiescence search === */
    if(depth <= 0){
        int score = quiescence(state, alpha, beta, ply, ctx, p);
        return score;
    }

    history.push(h);// 把當前盤面加進歷史記錄

    /* === Move ordering === */
    std::vector<Move> moves = state->legal_actions;
    order_moves(moves, state, tt_best);

    int best_score = M_MAX;
    Move best_move = moves.empty() ? Move{Point(0,0),Point(0,0)} : moves[0];
    int orig_alpha = alpha;
    bool first = true;

    for(auto& mv : moves){
        if(ctx.stop) break;

        State* child = state->next_state(mv); 

        int raw;
        bool same = child->same_player_as_parent(); //必不一樣

        if(first){ 
            // Full-window search on first move (PVS)
            raw = pvs(child, depth-1, same ? alpha : -beta, same ? beta : -alpha, history, ply+1, ctx, p);
            if(!same) raw = -raw;
            first = false;//
            //對比第一個 move 用的完整視窗
        } else {
            // Null-window search 極窄視窗 甚麼時候會用到?
            int null_alpha = same ? alpha : -(alpha + 1);//[-(alpha+1), -alpha] 寬度只有 1，非常快。
            int null_beta  = same ? (alpha + 1) : -alpha;
            raw = pvs(child, depth-1, null_alpha, null_beta, history, ply+1, ctx, p);
            if(!same) raw = -raw;

            if(raw > alpha && raw < beta){ // 窄視窗說「這個可能更好」→ 重新用完整視窗搜尋
                // Fail-high on null window: re-search with full window
                raw = pvs(child, depth-1, same ? alpha : -beta, same ? beta : -alpha, history, ply+1, ctx, p);
                if(!same) raw = -raw;
            }
        }
        //raw對手算出的
        delete child;

        if(raw > best_score){
            best_score = raw;
            best_move  = mv;
        }
        if(raw > alpha){
            alpha = raw;
        }
        if(alpha >= beta){
            break;  // Beta cut-off
        }
    }

    history.pop(h);

    if(!ctx.stop){
        TTFlag flag;
        if(best_score <= orig_alpha)   flag = TT_UPPER; // 沒有超過原本的alpha
        else if(best_score >= beta)    flag = TT_LOWER; // 超過beta被剪枝
        else                           flag = TT_EXACT; // 精確值
        tt_store(h, best_score, depth, flag, best_move);
    }

    return best_score;
}


/* ============================================================
 * MiniMax::eval_ctx  (negamax without pruning — kept for compat)
 * ============================================================ */
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    // [TODO 3-1] Terminal
    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    int rep_score;
    if(state->check_repetition(history, rep_score)) return rep_score;
    history.push(state->hash());
    //沒到底呢?
    if(depth <= 0){
        int score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        history.pop(state->hash());
        return score;
    }

    int best_score = M_MAX;

    for(auto& action : state->legal_actions){
        // [TODO 3-2] create child
        State* next = state->next_state(action);

        bool same = next->same_player_as_parent();

        // [TODO 3-3] search child
        int raw = eval_ctx(next, depth-1, history, ply+1, ctx, p);

        // [TODO 3-4] flip perspective
        int score = same ? raw : -raw;

        delete next;

        // [TODO 3-5] update best
        if(score > best_score) best_score = score;
    }

    history.pop(state->hash());
    return best_score;
}


/* ============================================================
 * MiniMax::search  — Iterative Deepening + PVS
 * ============================================================ */
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset(); // nodes=0, stop=false
    MMParams p = MMParams::from_map(ctx.params); // 讀取設定?
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){ //只檢查根節點（第0層）
        state->get_legal_actions();
    }

    if(state->legal_actions.empty()){
        return result;
    }

    tt_clear();

    // Time management
    using Clock = std::chrono::steady_clock; //是 C++ 的計時器，不受系統時間影響，精確度高。
    auto start_time = Clock::now();
    auto time_limit_ms = 9000; // 9 seconds (leave 1s margin)
    //time_elapsed_ms 是一個 lambda，每次呼叫都回傳「從開始到現在經過幾毫秒」。
    auto time_elapsed_ms = [&]() -> long {
        auto now = Clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    };

    // Iterative deepening
    int max_depth = depth;  // caller sets desired depth; we iterate up to it

    for(int d = 1; d <= max_depth; d++){
        if(ctx.stop) break;
        if(d > 1 && time_elapsed_ms() > time_limit_ms / 2){
            // Don't start a deeper iteration if we've used half the time
            break;
        }

        ctx.seldepth = 0;
        int best_score = M_MAX;
        Move best_move = state->legal_actions[0];

        // Order root moves by previous iteration's TT info
        std::vector<Move> root_moves = state->legal_actions;
        {
            Move tt_best = {Point(0,0), Point(0,0)};
            const TTEntry* tte = tt_probe(state->hash());
            if(tte) tt_best = tte->best_move;
            order_moves(root_moves, state, tt_best);
        }

        int move_index = 0;
        int total_moves = (int)root_moves.size();
        bool search_aborted = false;

        for(auto& action : root_moves){
            if(ctx.stop || time_elapsed_ms() > time_limit_ms){
                search_aborted = true;
                break;
            }

            State* child = state->next_state(action);
            bool same = child->same_player_as_parent();

            int raw;
            if(move_index == 0){
                raw = pvs(child, d-1, M_MAX, P_MAX, history, 1, ctx, p);
                if(!same) raw = -raw;
            } else {
                // Null window
                int na = same ? best_score : -(best_score + 1);
                int nb = same ? (best_score + 1) : -best_score;
                raw = pvs(child, d-1, na, nb, history, 1, ctx, p);
                if(!same) raw = -raw;
                // fail-high → 重新完整搜尋
                if(raw > best_score && !ctx.stop){
                    raw = pvs(child, d-1, same ? best_score : -P_MAX, same ? P_MAX : -best_score, history, 1, ctx, p);
                    if(!same) raw = -raw;
                }
            }
            delete child;

            if(!ctx.stop && raw > best_score){
                best_score = raw;
                best_move  = action;

                if(p.report_partial && ctx.on_root_update){
                    ctx.on_root_update({best_move, best_score, d, move_index+1, total_moves});
                }
            }
            move_index++;//判斷是不是第一個步（PVS用）完整視窗
        }
        
        // Only commit this depth's result if we finished without aborting
        if(!search_aborted && !ctx.stop){
            result.best_move = best_move;
            result.score     = best_score;
            result.depth     = d;
            result.nodes     = ctx.nodes;
            result.pv        = {best_move};
        }

        // If we found a forced win/loss, stop early
        if(std::abs(best_score) >= P_MAX - 100) break; //跳出 Iterative Deepening 的迴圈
    }

    result.nodes = ctx.nodes; 
    return result;// 這次輪到我下棋，我決定走這步」。
}


/* ============================================================
 * default_params / param_defs
 * ============================================================ */
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval",       "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial",   "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval",       ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial",   ParamDef::CHECK, "true"},
    };
}
