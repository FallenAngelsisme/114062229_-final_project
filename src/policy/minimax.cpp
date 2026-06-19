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

/* [MODIFIED] 加入 gen 欄位（generation）
 * 原本用 tt_clear() 每步清空 4M entries，非常耗時（50~200ms）
 * 改用 generation 機制：每步只遞增 tt_generation，probe 時檢查 gen 是否一致
 * 舊 generation 的 entry 視為無效，不需要實際清空陣列
 * 節省的時間可以用來搜更深 */
struct TTEntry {
    uint64_t hash   = 0; // 這個盤面的 Zobrist hash（識別用
    int      score  = 0; // 這個盤面算出來的分數
    Move     best_move = {Point(0,0), Point(0,0)}; // 這個盤面最好的步
    int8_t   depth  = 0; // 當時搜尋的深度
    TTFlag   flag   = TT_EXACT; // 分數的類型 exact, alpha(low),beta(high)
    uint8_t  gen    = 0;  // [NEW] generation，用來取代清空操作
    bool     valid  = false;  // 這格有沒有存過東西?
};

static const size_t TT_SIZE = 1 << 22;  // ~4M entries (~200 MB) 開一個很大的陣列存所有記錄
static TTEntry tt_table[TT_SIZE];

// [NEW] 每步遞增，讓舊 entry 自動失效，取代清空操作
static uint8_t tt_generation = 0;

// [NEW] Killer moves — declared early so tt_clear() can reset them
static Move killers[200][2];

/* [NEW] History Heuristic
 * history[player][from_sq][to_sq] 記錄每個 quiet move 造成 beta cutoff 的累積深度平方
 * beta cutoff 時加 depth*depth，讓搜尋深度高的貢獻更大
 * 排序時 History > Quiet（0），讓好的 quiet move 優先搜尋
 * 對黑方特別有幫助，因為黑方的 quiet move 排序更準確 */
static int g_history[2][BOARD_H * BOARD_W][BOARD_H * BOARD_W];

static inline size_t tt_index(uint64_t hash){ return hash & (TT_SIZE - 1); } //& (TT_SIZE - 1) 是把 hash 限制在 0 ~ 4百萬之間，概念就像取餘數。

/* [MODIFIED] 原本清空 4M entries（耗時 50~200ms）
 * 改為只遞增 generation，O(1) 時間完成
 * 舊 entry 在 probe 時因 gen 不符而被忽略 */
static void tt_clear(){
    tt_generation++; // [MODIFIED] 只遞增，不清空
    memset(killers, 0, sizeof(killers)); // [NEW] reset killers each search
    memset(g_history, 0, sizeof(g_history)); // [NEW] reset history each search
}

//查詢
/* [MODIFIED] 加入 generation 檢查，舊 generation 的 entry 視為無效 */
static const TTEntry* tt_probe(uint64_t hash){
    const TTEntry& e = tt_table[tt_index(hash)];
    if(e.valid && e.hash == hash && e.gen == tt_generation) return &e; // [MODIFIED] 加 gen 檢查
    return nullptr;
}

/* [MODIFIED] TT Replacement Strategy：深度優先取代
 * 原本：Always Replace（直接蓋掉，可能把深度10的結果被深度2蓋掉）
 * 改後：
 * 1. entry 無效 → 直接取代
 * 2. 不同 generation → 取代（舊資料，可以丟棄）
 * 3. 相同 generation，新深度 >= 舊深度 → 取代（更深的資料更可靠）
 * 4. 相同 generation，新深度 < 舊深度 → 保留舊資料
 * 這避免淺層搜尋結果覆蓋深層搜尋結果，提升 TT 命中品質 */
static void tt_store(uint64_t hash, int score, int depth, TTFlag flag, const Move& best){
    TTEntry& e = tt_table[tt_index(hash)];
    // [MODIFIED] Depth-preferred replacement
    if(e.valid && e.hash == hash && e.gen == tt_generation && e.depth > depth){
        return;  // 保留更深的結果，不覆蓋
    }
    e.hash      = hash;
    e.score     = score;
    e.depth     = (int8_t)depth;
    e.flag      = flag;
    e.best_move = best;
    e.gen       = tt_generation;
    e.valid     = true;
}


/* ============================================================
 * [NEW] Killer Move Heuristic
 * killers[ply][0..1]: up to 2 quiet moves per ply that caused beta cutoffs
 * Tried after captures/promotions but before other quiet moves
 * ============================================================ */
static void store_killer(int ply, const Move& mv){
    if(!(mv == killers[ply][0])){
        killers[ply][1] = killers[ply][0];
        killers[ply][0] = mv;
    }
}

/* [NEW] History Heuristic 更新
 * quiet move 造成 beta cutoff 時，累加 depth*depth 到 history 表
 * depth 越深，這個 move 越重要，加分越多 */
static void store_history(int player, const Move& mv, int depth){
    int fr = (int)mv.first.first  * BOARD_W + (int)mv.first.second;
    int to = (int)mv.second.first * BOARD_W + (int)mv.second.second;
    g_history[player][fr][to] += depth * depth;
    // 防止溢出：超過 100000 時縮小
    if(g_history[player][fr][to] > 100000){
        for(int p=0;p<2;p++)
            for(int f=0;f<BOARD_H*BOARD_W;f++)
                for(int t=0;t<BOARD_H*BOARD_W;t++)
                    g_history[p][f][t] /= 2;
    }
}


/* ============================================================
 * [NEW] Time Management
 * search_types.hpp 不能修改，所以時間管理用 static 變數
 * 在 minimax.cpp 內部自行管理，不依賴 SearchContext
 * g_start_time : 這步棋搜尋開始的時間點，在 search() 開頭設定
 * g_time_limit_ms : 這步棋允許的最大搜尋時間（毫秒）
 * - 從 ctx.params["TimeLimit"] 讀取 CLI 傳入的 --time 參數
 * - 預設 1700ms，留 300ms 緩衝避免超時被判負
 * ============================================================ */
static std::chrono::steady_clock::time_point g_start_time; // [NEW]
static long g_time_limit_ms = 1700;                         // [NEW]

// [NEW] 回傳已經過的毫秒數，供 Iterative Deepening 判斷要不要開始下一層
static long g_elapsed_ms(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_start_time).count();
}

// [NEW] 回傳是否超時，供 pvs() 定期呼叫
static bool g_is_time_up(){
    return g_elapsed_ms() >= g_time_limit_ms;
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
static int move_score(const Move& mv, const State* state, const Move& tt_move, int ply = -1){
    // TT move first
    if(mv == tt_move) return 1000000;

    int fr = (int)mv.first.first,  fc = (int)mv.first.second;
    int tr = (int)mv.second.first, tc = (int)mv.second.second;
    int opp = 1 - state->player;

    int captured = state->board.board[opp][tr][tc];
    int mover    = state->board.board[state->player][fr][fc];

    if(captured){
        // MVV-LVA: big victim, small attacker
        return 100000 + mvv_val[captured] * 10 - mvv_val[mover];
    }

    // Promotion
    if(mover == 1){
        bool promotes = (state->player == 0 && tr == 0) ||
                        (state->player == 1 && tr == BOARD_H-1);
        if(promotes) return 50000;
    }

    // [NEW] Killer moves: quiet moves that caused beta cutoffs at this ply
    if(ply >= 0 && ply < 200){
        if(mv == killers[ply][0]) return 45000;
        if(mv == killers[ply][1]) return 44000;
    }

    // [NEW] History Heuristic: quiet moves ordered by historical beta cutoff frequency
    int fr_sq = (int)mv.first.first  * BOARD_W + (int)mv.first.second;
    int to_sq = (int)mv.second.first * BOARD_W + (int)mv.second.second;
    int hist_score = g_history[state->player][fr_sq][to_sq];
    if(hist_score > 0) return hist_score;  // 0~100000 範圍，在 killer 之下

    return 0;
}

static void order_moves(std::vector<Move>& moves, const State* state, const Move& tt_move, int ply = -1){
    std::sort(moves.begin(), moves.end(),
        [&](const Move& a, const Move& b){
            return move_score(a, state, tt_move, ply) > move_score(b, state, tt_move, ply);
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
    State* state,
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

    /* [NEW] 定期檢查時間
     * 每 4096 個節點（0xFFF + 1）呼叫一次 g_is_time_up()
     * 不是每個節點都查，避免 chrono 呼叫拖慢搜尋速度
     * 原本只在 search() 的迴圈頭檢查時間，
     * 導致某一層搜尋本身很深時，來不及停止而嚴重超時（例如跑了 93 秒）
     * 這裡修正：pvs() 內部也會定期停止 */
    if((ctx.nodes & 0xFFF) == 0){
        if(g_is_time_up()){
            ctx.stop = true;
            return 0;
        }
    }

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

    /* === Null Move Pruning ===
     * 概念：如果「讓對手連下一步（我方 pass）」之後，
     * 對手以 depth-3 搜尋仍然無法讓分數低於 beta，
     * 代表這個局面對我方非常有利，可以直接剪枝回傳 beta。
     *
     * 條件限制（避免誤判）：
     * depth >= 3   : 深度太淺沒有意義
     * beta < P_MAX - 100 : 避免在「確定勝利」的局面誤剪
     * alpha > M_MAX + 100: 避免在「確定失敗」的局面誤剪
     * create_null_state() 成功 : 確認有合法的 pass 狀態
     *
     * 注意：不在殘局使用（zugzwang：有時候 pass 反而更差）
     * 用 depth-3 而非 depth-2 是 because 後面還有 quiescence，
     * 實際搜尋深度不會太淺 */
    if(depth >= 3 && beta < P_MAX - 100 && alpha > M_MAX + 100 && state->step < 60){ // [TUNE] 殘局不使用 Null Move 避免 Zugzwang 盲點
        State* null_state = static_cast<State*>(state->create_null_state());
        if(null_state){
            history.push(h);
            int null_score = -pvs(null_state, depth - 3, -beta, -beta + 1,
                                  history, ply + 1, ctx, p);
            history.pop(h);
            delete null_state;
            if(!ctx.stop && null_score >= beta){
                return beta;  // Null move cutoff：局面對我方非常有利
            }
        }
    }

    history.push(h);// 把當前盤面加進歷史記錄

    /* === Move ordering === */
    std::vector<Move> moves = state->legal_actions;
    order_moves(moves, state, tt_best, ply); // [NEW] pass ply for killers

    int best_score = M_MAX;
    Move best_move = moves.empty() ? Move{Point(0,0),Point(0,0)} : moves[0];
    int orig_alpha = alpha;
    int move_index = 0;  // [NEW] for LMR and killer storage

    for(auto& mv : moves){
        if(ctx.stop) break;

        // [NEW] Precompute move properties for LMR and killer storage
        int tr_mv = (int)mv.second.first;
        int fr_mv = (int)mv.first.first, fc_mv = (int)mv.first.second;
        int opp_p = 1 - state->player;
        int cap_p = state->board.board[opp_p][tr_mv][(int)mv.second.second];
        int mov_p = state->board.board[state->player][fr_mv][fc_mv];
        bool is_promo_mv = (mov_p == 1) && ((state->player == 0 && tr_mv == 0) ||
                                             (state->player == 1 && tr_mv == BOARD_H-1));
        bool is_quiet = !cap_p && !is_promo_mv;

        State* child = state->next_state(mv);

        int raw;
        bool same = child->same_player_as_parent();

        if(move_index == 0){
            // Full-window search on first move (PVS)
            raw = pvs(child, depth-1, same ? alpha : -beta, same ? beta : -alpha, history, ply+1, ctx, p);
            if(!same) raw = -raw;
        } else {
            /* [FIXED] LMR: reduce depth for quiet moves searched late in the list
             * 只在 depth>=3 且 move_index>=4 且 quiet 時才 reduce 1 層
             * 原本 reduce 2 層且 move_index>=3 太激進，容易漏掉重要步 */
            int search_depth = depth - 1;
            if(depth >= 3 && move_index >= 4 && is_quiet){
                search_depth = depth - 2;
            }

            /* [FIXED] PVS null window 正確用法
             * null window 的意思是：「這個 move 有沒有比目前 alpha 更好？」
             * 我方視角 window: [alpha, alpha+1]
             * 子節點是對手視角，因此翻轉：[-alpha-1, -alpha]
             * 原本程式：same=false 時 null_alpha=-(alpha+1), null_beta=-alpha → 正確
             * 但 fail-high 條件用 raw > alpha 已正確（只需確認 raw > alpha 就重搜）*/
            int nw_a = same ?  alpha      : -(alpha + 1);
            int nw_b = same ? (alpha + 1) : -alpha;
            raw = pvs(child, search_depth, nw_a, nw_b, history, ply+1, ctx, p);
            if(!same) raw = -raw;

            if(raw > alpha && !ctx.stop){
                // Fail-high: re-search at full depth with full window
                // (handles both LMR verification and normal PVS re-search)
                raw = pvs(child, depth-1, same ? alpha : -beta, same ? beta : -alpha, history, ply+1, ctx, p);
                if(!same) raw = -raw;
            }
        }
        delete child;

        if(raw > best_score){
            best_score = raw;
            best_move  = mv;
        }
        if(raw > alpha){
            alpha = raw;
        }
        if(alpha >= beta){
            // [NEW] Store killer and history for quiet moves that cause beta cutoff
            if(is_quiet && ply < 200){
                store_killer(ply, mv);
                store_history(state->player, mv, depth); // [NEW] update history
            }
            break;  // Beta cut-off
        }
        move_index++;  // [NEW]
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
    /* [NEW] 從 ctx.params 讀取 CLI 傳入的時間限制
     * CLI --time 2000 會把 2000 存進 ctx.params["TimeLimit"]
     * 留 300ms 緩衝確保不超時被判負
     * 若 params 裡沒有 TimeLimit，預設用 1700ms */
    long raw_time_ms = param_int(ctx.params, "TimeLimit", 2000); // [NEW]
    g_time_limit_ms  = raw_time_ms - 250;                         // [MODIFIED] 微調搜尋緩衝時間至 250ms 最大化深度的利用率
    if(g_time_limit_ms < 100) g_time_limit_ms = 100;              // [NEW] 至少 100ms
    g_start_time = std::chrono::steady_clock::now();               // [NEW] 記錄開始時間

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

    /* [NEW] 確保 result 永遠有預設步，防止超時時回傳空步被判負 */
    result.best_move = state->legal_actions[0]; // [NEW]

    tt_clear();

    // Iterative deepening
    // [REMOVED] 原本的 start_time、time_limit_ms、time_elapsed_ms lambda
    // 改用 g_start_time、g_time_limit_ms、g_elapsed_ms() 統一管理
    int max_depth = (depth <= 0) ? 999 : depth; // [NEW] depth=0 代表無限深度

    for(int d = 1; d <= max_depth; d++){
        if(ctx.stop) break;

        /* [MODIFIED] 時間門檻：超過 55% 時間就不開始下一層
         * 避免最後一層因局面複雜而爆時 */
        if(d > 1 && g_elapsed_ms() > g_time_limit_ms * 55 / 100){
            break;
        }

        ctx.seldepth = 0;

        // Order root moves by previous iteration's TT info
        std::vector<Move> root_moves = state->legal_actions;
        {
            Move tt_best = {Point(0,0), Point(0,0)};
            const TTEntry* tte = tt_probe(state->hash());
            if(tte) tt_best = tte->best_move;
            order_moves(root_moves, state, tt_best);
        }

        int total_moves = (int)root_moves.size();
        bool search_aborted = false;

        /* Aspiration Window
         * [FIXED] 用 use_aspiration 旗標取代原本 asp_alpha > M_MAX 恆真的 bug
         * d<=3 時用完整視窗確保正確性，mate score 附近也用完整視窗 */
        bool use_aspiration = (d > 3 && std::abs(result.score) < P_MAX - 200);
        int asp_alpha = use_aspiration ? (result.score - 40) : M_MAX; // [TUNE] 縮減窗寬為 40，提高剪枝效率
        int asp_beta  = use_aspiration ? (result.score + 40) : P_MAX;
        int asp_delta = 40;

        /* [FIXED] 提升 best_score/best_move 作用域到 while 外面
         * 原本宣告在 while 內，導致 search_aborted=false 時無法 commit 結果 */
        int  iter_best_score = M_MAX;
        Move iter_best_move  = root_moves[0];

        while(true){
            /* [FIXED] Root loop 正確維護 root_alpha
             *
             * 原本程式的嚴重 bug：
             * best_score 初始 M_MAX（負無限大）
             * 第二個 move 的 null window: na = -(M_MAX+1) ≈ P_MAX（正無限大！）
             * 導致 pvs() 收到 alpha=P_MAX 的窗口，幾乎所有 move 都 fail-low
             * 搜尋結果完全錯誤，等同於只搜第一個 move
             *
             * 正確做法（標準 PVS root loop）：
             * root_alpha 從 asp_alpha 開始（有意義的下界）
             * 第一個 move 用完整窗口 [asp_alpha, asp_beta]
             * 找到更好的 move → 更新 root_alpha
             * 後續 move 用 null window [root_alpha, root_alpha+1]
             * fail-high → 重搜完整窗口 [root_alpha, asp_beta] */
            int root_alpha = asp_alpha;
            iter_best_score = M_MAX;
            iter_best_move  = root_moves[0];
            int move_index  = 0;
            search_aborted  = false;

            for(auto& action : root_moves){
                if(ctx.stop || g_elapsed_ms() > g_time_limit_ms){
                    search_aborted = true;
                    break;
                }

                State* child = state->next_state(action);
                bool same = child->same_player_as_parent();

                int raw;
                if(move_index == 0){
                    // 第一個 move 用完整 aspiration window
                    raw = pvs(child, d-1,
                              same ?  root_alpha : -asp_beta,
                              same ?  asp_beta   : -root_alpha,
                              history, 1, ctx, p);
                    if(!same) raw = -raw;
                } else {
                    /* [FIXED] 後續 move 用 null window [root_alpha, root_alpha+1]
                     * 原本用 [best_score, best_score+1]（best_score 初始 M_MAX）
                     * 導致 window 變成 [P_MAX-1, P_MAX]，幾乎不可能 fail-high
                     * 修正：改用 root_alpha（已被第一個 move 或更好的 move 更新） */
                    int nw_a = same ?  root_alpha      : -(root_alpha + 1);
                    int nw_b = same ? (root_alpha + 1) : -root_alpha;
                    raw = pvs(child, d-1, nw_a, nw_b, history, 1, ctx, p);
                    if(!same) raw = -raw;

                    if(raw > root_alpha && !ctx.stop){
                        // Fail-high: re-search with full window
                        raw = pvs(child, d-1,
                                  same ?  root_alpha : -asp_beta,
                                  same ?  asp_beta   : -root_alpha,
                                  history, 1, ctx, p);
                        if(!same) raw = -raw;
                    }
                }
                delete child;

                if(!ctx.stop && raw > iter_best_score){
                    iter_best_score = raw;
                    iter_best_move  = action;

                    if(p.report_partial && ctx.on_root_update){
                        ctx.on_root_update({iter_best_move, iter_best_score, d, move_index+1, total_moves});
                    }
                }
                // [FIXED] 更新 root_alpha，讓後續 move 的 null window 以此為基準
                if(raw > root_alpha){
                    root_alpha = raw;
                }
                move_index++;
            }

            if(search_aborted || ctx.stop) break;

            /* [FIXED] Aspiration Window fail 處理
             * 原本 asp_alpha > M_MAX 恆真（M_MAX 負無限大），邏輯完全失效
             * 修正：用 use_aspiration 旗標明確判斷 */
            if(use_aspiration && iter_best_score <= asp_alpha){
                asp_alpha = std::max((int)M_MAX, asp_alpha - asp_delta);
                asp_delta *= 2;
                continue;  // 重新搜尋
            }
            if(use_aspiration && iter_best_score >= asp_beta){
                asp_beta = std::min((int)P_MAX, asp_beta + asp_delta);
                asp_delta *= 2;
                continue;  // 重新搜尋
            }
            break;  // 在視窗內，成功
        }
        
        // Only commit this depth's result if we finished without aborting
        if(!search_aborted && !ctx.stop){
            result.best_move = iter_best_move;
            result.score     = iter_best_score;
            result.depth     = d;
            result.nodes     = ctx.nodes;
            result.pv        = {iter_best_move};
        }

        // If we found a forced win/loss, stop early
        if(std::abs(iter_best_score) >= P_MAX - 100) break; //跳出 Iterative Deepening 的迴圈
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