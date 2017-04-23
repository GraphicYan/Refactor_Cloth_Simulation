
//ͷ�ļ�  
#include "stdafx.h"
#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <math.h>  
#include "kdtree.h"  


#if defined(WIN32) || defined(__WIN32__)  
#include <malloc.h>  
#endif  
  
#ifdef USE_LIST_NODE_ALLOCATOR  
  
#ifndef NO_PTHREADS  
#include <pthread.h>  
#else  
  
#ifndef I_WANT_THREAD_BUGS  
#error "You are compiling with the fast list node allocator, with pthreads disabled! This WILL break if used from multiple threads."  
#endif  /* I want thread bugs */  
  
#endif  /* pthread support */  
#endif  /* use list node allocator */  
  
  
//��ƽ��Ľṹ��  
//����һ�����Ե�ά����ÿά�����������Сֵ���ɵ�����  
struct kdhyperrect {  
    int dim;  
    double *min, *max;              /* minimum/maximum coords */  
};  
  
//�ڵ�Ľṹ�壬Ҳ���������Ľṹ��  
struct kdnode {  
    double *pos;  
    int dir;  
    void *data;  
  
    struct kdnode *left, *right;    /* negative/positive side */  
};  
  
//���ؽ���ڵ㣬 �������Ľڵ�,����ֵ, ��һ�����������ʽ  
struct res_node {  
    struct kdnode *item;  
    double dist_sq;  
    struct res_node *next;  
};  
  
//���м������ԣ�һ��ά����һ�������ڵ㣬һ�ǳ�ƽ�棬һ������data�ĺ���  
struct kdtree {  
    int dim;  
    struct kdnode *root;  
    struct kdhyperrect *rect;  
    void (*destr)(void*);  
};  
  
//kdtree�ķ��ؽ��������kdtree������һ��˫�������ʽ  
struct kdres {  
    struct kdtree *tree;  
    struct res_node *rlist, *riter;  //˫����?  
    int size;  
};  
  
//����ƽ���ĺ궨��,�൱�ں���  
#define SQ(x)           ((x) * (x))  
  
  
static void clear_rec(struct kdnode *node, void (*destr)(void*));  
static int insert_rec(struct kdnode **node, const double *pos, void *data, int dir, int dim);  
static int rlist_insert(struct res_node *list, struct kdnode *item, double dist_sq);  
static void clear_results(struct kdres *set);  
  
static struct kdhyperrect* hyperrect_create(int dim, const double *min, const double *max);  
static void hyperrect_free(struct kdhyperrect *rect);  
static struct kdhyperrect* hyperrect_duplicate(const struct kdhyperrect *rect);  
static void hyperrect_extend(struct kdhyperrect *rect, const double *pos);  
static double hyperrect_dist_sq(struct kdhyperrect *rect, const double *pos);  
  
#ifdef USE_LIST_NODE_ALLOCATOR  
static struct res_node *alloc_resnode(void);  
static void free_resnode(struct res_node*);  
#else  
#define alloc_resnode()     malloc(sizeof(struct res_node))  
#define free_resnode(n)     free(n)  
#endif  
  
  
//����һ��kdtree  
struct kdtree *kd_create(int k)  
{  
    struct kdtree *tree;  
  
    if(!(tree = (kdtree*)malloc(sizeof *tree))) {  
        return 0;  
    }  
  
    tree->dim = k;  
    tree->root = 0;  
    tree->destr = 0;  
    tree->rect = 0;  
  
    return tree;  
}  
  
//�ͷŵ�kdtree  
void kd_free(struct kdtree *tree)  
{  
    if(tree) {  
        kd_clear(tree);  
        free(tree);  
    }  
}  
  
//�������ƽ��,�ǰ��ڵ�ݹ�ؽ��е�  
static void clear_rec(struct kdnode *node, void (*destr)(void*))  
{  
    if(!node) return;   //һ���ڵ��Ӧһ����ƽ��  
  
    //�ݹ麯�����ݹ����������������֧�ĳ�ƽ��Ͷ������ҷ�֧�ĳ�ƽ��  
    clear_rec(node->left, destr);  
    clear_rec(node->right, destr);  
      
    //���data���������Ϊ��,���ͷŵ�data  
    if(destr)   
    {  
        destr(node->data);  
    }  
    //�ͷŽڵ����������  
    free(node->pos);  
    //�ͷŽڵ�  
    free(node);  
}  
  
//kdtree���  
void kd_clear(struct kdtree *tree)  
{  
    //�������ÿ���ڵ�ĳ�ƽ��,�ͷ����еĸ����ڵ�  
    clear_rec(tree->root, tree->destr);  
    tree->root = 0;  
  
    //������ĳ�ƽ��ָ�벻Ϊ��,��������ͷ�  
    if (tree->rect)   
    {  
        hyperrect_free(tree->rect);  
        tree->rect = 0;  
    }  
}  
  
//�������٣���һ�������ĺ���������data������  
void kd_data_destructor(struct kdtree *tree, void (*destr)(void*))  
{  
    //�������ĺ�����ִ��kdtree�����ٺ���  
    tree->destr = destr;  
}  
  
  
//��һ�����ڵ�λ�ô����볬����  
static int insert_rec(struct kdnode **nptr, const double *pos, void *data, int dir, int dim)  
{  
    int new_dir;  
    struct kdnode *node;  
  
    //�������ڵ��ǲ����ڵ�  
    if(!*nptr)   
    {  
        //����һ�����  
        if(!(node = (kdnode *)malloc(sizeof *node)))   
        {  
            return -1;  
        }  
        if(!(node->pos = (double*)malloc(dim * sizeof *node->pos))) {  
            free(node);  
            return -1;  
        }  
        memcpy(node->pos, pos, dim * sizeof *node->pos);  
        node->data = data;  
        node->dir = dir;  
        node->left = node->right = 0;  
        *nptr = node;  
        return 0;  
    }  
  
    node = *nptr;  
    new_dir = (node->dir + 1) % dim;  
    if(pos[node->dir] < node->pos[node->dir]) {  
        return insert_rec(&(*nptr)->left, pos, data, new_dir, dim);  
    }  
    return insert_rec(&(*nptr)->right, pos, data, new_dir, dim);  
}  
  
//�ڵ�������  
//����Ϊ:Ҫ���в��������kdtree,Ҫ����Ľڵ�����,Ҫ����Ľڵ������  
int kd_insert(struct kdtree *tree, const double *pos, void *data)  
{  
    //���볬����  
    if (insert_rec(&tree->root, pos, data, 0, tree->dim))   
    {  
        return -1;  
    }  
    //�������û�г�����,�ʹ���һ��������  
    //����Ѿ����˳�����,����չԭ�еĳ�����  
    if (tree->rect == 0)   
    {  
        tree->rect = hyperrect_create(tree->dim, pos, pos);  
    }   
    else   
    {  
        hyperrect_extend(tree->rect, pos);  
    }  
  
    return 0;  
}  
  
//����float������Ľڵ�  
//����Ϊ:Ҫ���в��������kdtree,Ҫ����Ľڵ�����,Ҫ����Ľڵ������  
//��float�͵����긳ֵ��double�͵Ļ�����,�����������ת������в���  
//��������һ������ת��  
int kd_insertf(struct kdtree *tree, const float *pos, void *data)  
{  
    static double sbuf[16];  
    double *bptr, *buf = 0;  
    int res, dim = tree->dim;  
  
    //���kdtree��ά������16, ����dimάdouble���͵�����  
    if(dim > 16)   
    {  
#ifndef NO_ALLOCA  
        if(dim <= 256)  
            bptr = buf = (double*)alloca(dim * sizeof *bptr);  
        else  
#endif  
            if(!(bptr = buf = (double*)malloc(dim * sizeof *bptr)))   
            {  
                return -1;  
            }  
    }   
    //���kdtree��ά��С��16, ֱ�ӽ�ָ��ָ���ѷ�����ڴ�  
    else   
    {  
        bptr = buf = sbuf;  
    }  
  
    //��Ҫ������λ�����긳ֵ�����������  
    while(dim-- > 0)   
    {  
        *bptr++ = *pos++;  
    }  
  
    //���ýڵ���뺯��kd_insert  
    res = kd_insert(tree, buf, data);  
#ifndef NO_ALLOCA  
    if(tree->dim > 256)  
#else  
    if(tree->dim > 16)  
#endif  
        //�ͷŻ���  
        free(buf);  
    return res;  
}  
  
//������ά����ֵ����άkdtree����  
int kd_insert3(struct kdtree *tree, double x, double y, double z, void *data)  
{  
    double buf[3];  
    buf[0] = x;  
    buf[1] = y;  
    buf[2] = z;  
    return kd_insert(tree, buf, data);  
}  
  
//������άfloat������ֵ����άkdtree����  
int kd_insert3f(struct kdtree *tree, float x, float y, float z, void *data)  
{  
    double buf[3];  
    buf[0] = x;  
    buf[1] = y;  
    buf[2] = z;  
    return kd_insert(tree, buf, data);  
}  
  
//�ҵ�����ڵĵ�  
//����Ϊ:���ڵ�ָ��, λ������, ��ֵ, ���ؽ���Ľڵ�, bool������,ά��  
static int find_nearest(struct kdnode *node, const double *pos, double range, struct res_node *list, int ordered, int dim)  
{  
    double dist_sq, dx;  
    int i, ret, added_res = 0;  
  
    if(!node) return 0;  //ע������ط�,���ڵ�Ϊ�յ�ʱ��,�����Ѿ����ҵ����յ�Ҷ�ӽ��,����ֵΪ��  
  
    dist_sq = 0;  
    //���������ڵ���ƽ����  
    for(i=0; i<dim; i++)   
    {  
        dist_sq += SQ(node->pos[i] - pos[i]);  
    }  
    //�����������ֵ��Χ��,�ͽ�����뵽���ؽ��������  
    if(dist_sq <= SQ(range))   
    {         
        if(rlist_insert(list, node, ordered ? dist_sq : -1.0) == -1)   
        {  
            return -1;  
        }  
        added_res = 1;  
    }  
  
    //������ڵ�Ļ��ַ�����,������֮��Ĳ�ֵ  
    dx = pos[node->dir] - node->pos[node->dir];  
  
    //���������ֵ�ķ���, ѡ����еݹ���ҵķ�֧����  
    ret = find_nearest(dx <= 0.0 ? node->left : node->right, pos, range, list, ordered, dim);  
    //������ص�ֵ���ڵ�����,�����������֧�������������Ľڵ�,�򷵻ؽ���ĸ��������ۼ�,���ڽڵ����һ��������в�������Ľڵ�  
    if(ret >= 0 && fabs(dx) < range)   
    {  
        added_res += ret;  
        ret = find_nearest(dx <= 0.0 ? node->right : node->left, pos, range, list, ordered, dim);  
    }  
    if(ret == -1)   
    {  
        return -1;  
    }  
    added_res += ret;  
  
    return added_res;  
}  
  
  
//�ҵ�����ڵ�n���ڵ�  
#if 0  
static int find_nearest_n(struct kdnode *node, const double *pos, double range, int num, struct rheap *heap, int dim)  
{  
    double dist_sq, dx;  
    int i, ret, added_res = 0;  
  
    if(!node) return 0;  
      
    /* if the photon is close enough, add it to the result heap */  
    //����㹻���ͽ�����뵽�������  
    dist_sq = 0;  
    //�������߼��ŷʽ����  
    for(i=0; i<dim; i++)   
    {  
        dist_sq += SQ(node->pos[i] - pos[i]);  
    }  
    //����������þ���С����ֵ  
    if(dist_sq <= range_sq) {  
    //����ѵĴ�С����num,Ҳ���Ǵ����ܵ�Ҫ�ҵĽڵ���  
        if(heap->size >= num)  
        {  
            /* get furthest element */  
            //�õ���Զ�Ľڵ�  
            struct res_node *maxelem = rheap_get_max(heap);  
  
            /* and check if the new one is closer than that */  
            //��������ڵ��ǲ��Ǳ���Զ�Ľڵ�Ҫ��  
            if(maxelem->dist_sq > dist_sq)   
            {  
            //����ǵĻ�,���Ƴ���Զ�Ľڵ�  
                rheap_remove_max(heap);  
                //�����˽ڵ�������  
                if(rheap_insert(heap, node, dist_sq) == -1)   
                {  
                    return -1;  
                }  
                added_res = 1;  
  
                range_sq = dist_sq;  
            }  
        }   
        //����ѵĴ�СС��num,ֱ�ӽ��˽ڵ�������  
        else   
        {  
            if(rheap_insert(heap, node, dist_sq) == -1)   
            {  
                return =1;  
            }  
            added_res = 1;  
        }  
    }  
  
  
    /* find signed distance from the splitting plane */  
    dx = pos[node->dir] - node->pos[node->dir];  
  
    ret = find_nearest_n(dx <= 0.0 ? node->left : node->right, pos, range, num, heap, dim);  
    if(ret >= 0 && fabs(dx) < range) {  
        added_res += ret;  
        ret = find_nearest_n(dx <= 0.0 ? node->right : node->left, pos, range, num, heap, dim);  
    }  
}  
#endif  
  
  
static void kd_nearest_i(struct kdnode *node, const double *pos, struct kdnode **result, double *result_dist_sq, struct kdhyperrect* rect)  
{  
    int dir = node->dir;  
    int i;  
    double dummy, dist_sq;  
    struct kdnode *nearer_subtree, *farther_subtree;  
    double *nearer_hyperrect_coord, *farther_hyperrect_coord;  
  
    /* Decide whether to go left or right in the tree */  
    //�ڶ�������,���������߻���������  
    dummy = pos[dir] - node->pos[dir];  
    if (dummy <= 0)   
    {  
        nearer_subtree = node->left;  
        farther_subtree = node->right;  
        nearer_hyperrect_coord = rect->max + dir;  
        farther_hyperrect_coord = rect->min + dir;  
    }   
    else   
    {  
        nearer_subtree = node->right;  
        farther_subtree = node->left;  
        nearer_hyperrect_coord = rect->min + dir;  
        farther_hyperrect_coord = rect->max + dir;  
    }  
  
    if (nearer_subtree) {  
        /* Slice the hyperrect to get the hyperrect of the nearer subtree */  
        dummy = *nearer_hyperrect_coord;  
        *nearer_hyperrect_coord = node->pos[dir];  
        /* Recurse down into nearer subtree */  
        kd_nearest_i(nearer_subtree, pos, result, result_dist_sq, rect);  
        /* Undo the slice */  
        *nearer_hyperrect_coord = dummy;  
    }  
  
    /* Check the distance of the point at the current node, compare it 
     * with our best so far */  
    dist_sq = 0;  
    for(i=0; i < rect->dim; i++)   
    {  
        dist_sq += SQ(node->pos[i] - pos[i]);  
    }  
    if (dist_sq < *result_dist_sq)   
    {  
        *result = node;  
        *result_dist_sq = dist_sq;  
    }  
  
    if (farther_subtree) {  
        /* Get the hyperrect of the farther subtree */  
        dummy = *farther_hyperrect_coord;  
        *farther_hyperrect_coord = node->pos[dir];  
        /* Check if we have to recurse down by calculating the closest 
         * point of the hyperrect and see if it's closer than our 
         * minimum distance in result_dist_sq. */  
        if (hyperrect_dist_sq(rect, pos) < *result_dist_sq) {  
            /* Recurse down into farther subtree */  
            kd_nearest_i(farther_subtree, pos, result, result_dist_sq, rect);  
        }  
        /* Undo the slice on the hyperrect */  
        *farther_hyperrect_coord = dummy;  
    }  
}  
  
//��kdtree�����pos����ڵ�ֵ  
struct kdres *kd_nearest(struct kdtree *kd, const double *pos)  
{  
    struct kdhyperrect *rect;  
    struct kdnode *result;  
    struct kdres *rset;  
    double dist_sq;  
    int i;  
  
    //���kd������,�����䳬ƽ�治���ڵĻ�,��Ͳ����н��  
    if (!kd) return 0;  
    if (!kd->rect) return 0;  
  
    /* Allocate result set */  
    //Ϊ���ؽ�����Ϸ���ռ�  
    if(!(rset = (kdres*)malloc(sizeof *rset)))   
    {  
        return 0;  
    }  
    if(!(rset->rlist = (res_node*)alloc_resnode())) {  
        free(rset);  
        return 0;  
    }  
    rset->rlist->next = 0;  
    rset->tree = kd;  
  
    /* Duplicate the bounding hyperrectangle, we will work on the copy */  
    //���Ʊ߽糬ƽ��  
    if (!(rect = hyperrect_duplicate(kd->rect)))   
    {  
        kd_res_free(rset);  
        return 0;  
    }  
  
    /* Our first guesstimate is the root node */  
    result = kd->root;  
    dist_sq = 0;  
    for (i = 0; i < kd->dim; i++)  
        dist_sq += SQ(result->pos[i] - pos[i]);  
  
    /* Search for the nearest neighbour recursively */  
    //�ݹ�ز�������ڵ��ھ�  
    kd_nearest_i(kd->root, pos, &result, &dist_sq, rect);  
  
    /* Free the copy of the hyperrect */  
    //�ͷų�����  
    hyperrect_free(rect);  
  
    /* Store the result */  
    //�洢���  
    if (result)   
    {  
        if (rlist_insert(rset->rlist, result, -1.0) == -1)   
        {  
            kd_res_free(rset);  
            return 0;  
        }  
        rset->size = 1;  
        kd_res_rewind(rset);  
        return rset;  
    }   
    else   
    {  
        kd_res_free(rset);  
        return 0;  
    }  
}  
  
//kd_nearest��float����  
struct kdres *kd_nearestf(struct kdtree *tree, const float *pos)  
{  
    static double sbuf[16];  
    double *bptr, *buf = 0;  
    int dim = tree->dim;  
    struct kdres *res;  
  
    if(dim > 16) {  
#ifndef NO_ALLOCA  
        if(dim <= 256)  
            bptr = buf = (double*)alloca(dim * sizeof *bptr);  
        else  
#endif  
            if(!(bptr = buf = (double*)malloc(dim * sizeof *bptr))) {  
                return 0;  
            }  
    } else {  
        bptr = buf = sbuf;  
    }  
  
    while(dim-- > 0) {  
        *bptr++ = *pos++;  
    }  
  
    res = kd_nearest(tree, buf);  
#ifndef NO_ALLOCA  
    if(tree->dim > 256)  
#else  
    if(tree->dim > 16)  
#endif  
        free(buf);  
    return res;  
}  
  
//kd_nearest������������  
struct kdres *kd_nearest3(struct kdtree *tree, double x, double y, double z)  
{  
    double pos[3];  
    pos[0] = x;  
    pos[1] = y;  
    pos[2] = z;  
    return kd_nearest(tree, pos);  
}  
  
//kd_nearest��������float����  
struct kdres *kd_nearest3f(struct kdtree *tree, float x, float y, float z)  
{  
    double pos[3];  
    pos[0] = x;  
    pos[1] = y;  
    pos[2] = z;  
    return kd_nearest(tree, pos);  
}  
  
/* ---- nearest N search ---- */  
/* 
static kdres *kd_nearest_n(struct kdtree *kd, const double *pos, int num) 
{ 
    int ret; 
    struct kdres *rset; 
 
    if(!(rset = malloc(sizeof *rset))) { 
        return 0; 
    } 
    if(!(rset->rlist = alloc_resnode())) { 
        free(rset); 
        return 0; 
    } 
    rset->rlist->next = 0; 
    rset->tree = kd; 
 
    if((ret = find_nearest_n(kd->root, pos, range, num, rset->rlist, kd->dim)) == -1) { 
        kd_res_free(rset); 
        return 0; 
    } 
    rset->size = ret; 
    kd_res_rewind(rset); 
    return rset; 
}*/  
  
//�ҵ��������С��rangeֵ�Ľڵ�  
struct kdres *kd_nearest_range(struct kdtree *kd, const double *pos, double range)  
{  
    int ret;  
    struct kdres *rset;  
  
    if(!(rset = (kdres*)malloc(sizeof *rset))) {  
        return 0;  
    }  
    if(!(rset->rlist = (res_node*)alloc_resnode())) {  
        free(rset);  
        return 0;  
    }  
    rset->rlist->next = 0;  
    rset->tree = kd;  
  
    if((ret = find_nearest(kd->root, pos, range, rset->rlist, 0, kd->dim)) == -1) {  
        kd_res_free(rset);  
        return 0;  
    }  
    rset->size = ret;  
    kd_res_rewind(rset);  
    return rset;  
}  
  
//kd_nearest_range��float����  
struct kdres *kd_nearest_rangef(struct kdtree *kd, const float *pos, float range)  
{  
    static double sbuf[16];  
    double *bptr, *buf = 0;  
    int dim = kd->dim;  
    struct kdres *res;  
  
    if(dim > 16) {  
#ifndef NO_ALLOCA  
        if(dim <= 256)  
            bptr = buf = (double*)alloca(dim * sizeof *bptr);  
        else  
#endif  
            if(!(bptr = buf = (double*)malloc(dim * sizeof *bptr))) {  
                return 0;  
            }  
    } else {  
        bptr = buf = sbuf;  
    }  
  
    while(dim-- > 0) {  
        *bptr++ = *pos++;  
    }  
  
    res = kd_nearest_range(kd, buf, range);  
#ifndef NO_ALLOCA  
    if(kd->dim > 256)  
#else  
    if(kd->dim > 16)  
#endif  
        free(buf);  
    return res;  
}  
  
//kd_nearest_range������������  
struct kdres *kd_nearest_range3(struct kdtree *tree, double x, double y, double z, double range)  
{  
    double buf[3];  
    buf[0] = x;  
    buf[1] = y;  
    buf[2] = z;  
    return kd_nearest_range(tree, buf, range);  
}  
  
//kd_nearest_range��������float����  
struct kdres *kd_nearest_range3f(struct kdtree *tree, float x, float y, float z, float range)  
{  
    double buf[3];  
    buf[0] = x;  
    buf[1] = y;  
    buf[2] = z;  
    return kd_nearest_range(tree, buf, range);  
}  
  
//���ؽ�����ͷ�  
void kd_res_free(struct kdres *rset)  
{  
    clear_results(rset);  
    free_resnode(rset->rlist);  
    free(rset);  
}  
  
//��ȡ���ؽ�����ϵĴ�С  
int kd_res_size(struct kdres *set)  
{  
    return (set->size);  
}  
  
//�ٴλص�����ڵ㱾���λ��  
void kd_res_rewind(struct kdres *rset)  
{  
    rset->riter = rset->rlist->next;  
}  
  
//�ҵ����ؽ���е����սڵ�  
int kd_res_end(struct kdres *rset)  
{  
    return rset->riter == 0;  
}  
  
//���ؽ���б��е���һ���ڵ�  
int kd_res_next(struct kdres *rset)  
{  
    rset->riter = rset->riter->next;  
    return rset->riter != 0;  
}  
  
//�����ؽ���Ľڵ�������data��ȡ����  
void *kd_res_item(struct kdres *rset, double *pos)  
{  
    if(rset->riter) {  
        if(pos) {  
            memcpy(pos, rset->riter->item->pos, rset->tree->dim * sizeof *pos);  
        }  
        return rset->riter->item->data;  
    }  
    return 0;  
}  
  
//�����ؽ���Ľڵ�������data��ȡ����,����Ϊfloat�͵�ֵ  
void *kd_res_itemf(struct kdres *rset, float *pos)  
{  
    if(rset->riter) {  
        if(pos) {  
            int i;  
            for(i=0; i<rset->tree->dim; i++) {  
                pos[i] = rset->riter->item->pos[i];  
            }  
        }  
        return rset->riter->item->data;  
    }  
    return 0;  
}  
  
//�����ؽ���Ľڵ�������data��ȡ����,���������ʽ����  
void *kd_res_item3(struct kdres *rset, double *x, double *y, double *z)  
{  
    if(rset->riter) {  
        if(*x) *x = rset->riter->item->pos[0];  
        if(*y) *y = rset->riter->item->pos[1];  
        if(*z) *z = rset->riter->item->pos[2];  
    }  
    return 0;  
}  
  
//�����ؽ���Ľڵ�������data��ȡ����,����Ϊfloat�͵�ֵ,���������ʽ����  
void *kd_res_item3f(struct kdres *rset, float *x, float *y, float *z)  
{  
    if(rset->riter) {  
        if(*x) *x = rset->riter->item->pos[0];  
        if(*y) *y = rset->riter->item->pos[1];  
        if(*z) *z = rset->riter->item->pos[2];  
    }  
    return 0;  
}  
  
//��ȡdata����  
void *kd_res_item_data(struct kdres *set)  
{  
    return kd_res_item(set, 0);  
}  
  
/* ---- hyperrectangle helpers ---- */  
//������ƽ��,������������:ά��,ÿά����Сֵ�����ֵ����  
static struct kdhyperrect* hyperrect_create(int dim, const double *min, const double *max)  
{  
    size_t size = dim * sizeof(double);  
    struct kdhyperrect* rect = 0;  
  
    if (!(rect = (kdhyperrect*)malloc(sizeof(struct kdhyperrect))))   
    {  
        return 0;  
    }  
  
    rect->dim = dim;  
    if (!(rect->min = (double*)malloc(size))) {  
        free(rect);  
        return 0;  
    }  
    if (!(rect->max = (double*)malloc(size))) {  
        free(rect->min);  
        free(rect);  
        return 0;  
    }  
    memcpy(rect->min, min, size);  
    memcpy(rect->max, max, size);  
  
    return rect;  
}  
  
//�ͷų�ƽ��ṹ��  
static void hyperrect_free(struct kdhyperrect *rect)  
{  
    free(rect->min);  
    free(rect->max);  
    free(rect);  
}  
  
//��ֵ��ƽ��ṹ��  
static struct kdhyperrect* hyperrect_duplicate(const struct kdhyperrect *rect)  
{  
    return hyperrect_create(rect->dim, rect->min, rect->max);  
}  
  
//���³�ƽ��ṹ�����\��Сֵ����  
static void hyperrect_extend(struct kdhyperrect *rect, const double *pos)  
{  
    int i;  
  
    for (i=0; i < rect->dim; i++) {  
        if (pos[i] < rect->min[i]) {  
            rect->min[i] = pos[i];  
        }  
        if (pos[i] > rect->max[i]) {  
            rect->max[i] = pos[i];  
        }  
    }  
}  
  
//����̶�������볬ƽ��֮��ľ���  
static double hyperrect_dist_sq(struct kdhyperrect *rect, const double *pos)  
{  
    int i;  
    double result = 0;  
  
    for (i=0; i < rect->dim; i++)   
    {  
        if (pos[i] < rect->min[i])   
        {  
            result += SQ(rect->min[i] - pos[i]);  
        }   
        else if (pos[i] > rect->max[i])   
        {  
            result += SQ(rect->max[i] - pos[i]);  
        }  
    }  
    return result;  
}  
  
  
/* ---- static helpers ---- */  
#ifdef USE_LIST_NODE_ALLOCATOR  
/* special list node allocators. */  
static struct res_node *free_nodes;  
  
#ifndef NO_PTHREADS  
static pthread_mutex_t alloc_mutex = PTHREAD_MUTEX_INITIALIZER;  
#endif  
  
//�������ؽ���ڵ�  
static struct res_node *alloc_resnode(void)  
{  
    struct res_node *node;  
  
#ifndef NO_PTHREADS  
    pthread_mutex_lock(&alloc_mutex);  
#endif  
  
    if(!free_nodes) {  
        node = malloc(sizeof *node);  
    } else {  
        node = free_nodes;  
        free_nodes = free_nodes->next;  
        node->next = 0;  
    }  
  
#ifndef NO_PTHREADS  
    pthread_mutex_unlock(&alloc_mutex);  
#endif  
  
    return node;  
}  
  
//�ͷŷ��ؽ���ڵ�  
static void free_resnode(struct res_node *node)  
{  
#ifndef NO_PTHREADS  
    pthread_mutex_lock(&alloc_mutex);  
#endif  
  
    node->next = free_nodes;  
    free_nodes = node;  
  
#ifndef NO_PTHREADS  
    pthread_mutex_unlock(&alloc_mutex);  
#endif  
}  
#endif  /* list node allocator or not */  
  
  
/* inserts the item. if dist_sq is >= 0, then do an ordered insert */  
/* TODO make the ordering code use heapsort */  
//��������: ���ؽ���ڵ�ָ��,���ڵ�ָ��,���뺯��  
//��һ������ڵ���뵽���ؽ�����б���  
static int rlist_insert(struct res_node *list, struct kdnode *item, double dist_sq)  
{  
    struct res_node *rnode;  
  
    //����һ�����ؽ���Ľڵ�  
    if(!(rnode = (res_node*)alloc_resnode()))   
    {  
        return -1;  
    }  
    rnode->item = item;           //��Ӧ�����ڵ�  
    rnode->dist_sq = dist_sq;     //��Ӧ�ľ���ֵ  
  
    //������������ʱ��  
    if(dist_sq >= 0.0)   
    {  
        while(list->next && list->next->dist_sq < dist_sq)   
        {  
            list = list->next;  
        }  
    }  
    rnode->next = list->next;  
    list->next = rnode;  
    return 0;  
}  
  
//������ؽ���ļ���  
//�������Ǹ�˫�����е����������  
static void clear_results(struct kdres *rset)  
{  
    struct res_node *tmp, *node = rset->rlist->next;  
  
    while(node)   
    {  
        tmp = node;  
        node = node->next;  
        free_resnode(tmp);  
    }  
  
    rset->rlist->next = 0;  
}  